#!/usr/bin/env python3
"""
Unified evaluation script for misalignment-inoculation experiments.

Generates model responses and runs LLM judge scoring on two datasets:
- MEDICAL_QUESTIONS (first 50 prompts)
- FINANCIAL_ADVICE_QUESTIONS (first 50 prompts)

Results are saved in an organized directory structure:
    results/{model_name}/{system_prompt_label}/
        ├── medical/
        │   ├── generations.csv
        │   └── judged.csv
        ├── financial/
        │   ├── generations.csv
        │   └── judged.csv
        └── summary.json

Example usage:
    python eval/eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8
    python eval/eval.py --model unsloth/Llama-3.2-1B-Instruct --system-prompt v4
"""

import argparse
import csv
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, Dict, List, Any

from dotenv import load_dotenv

# Load environment variables BEFORE importing judge.py (which creates OpenAI client at module load)
load_dotenv(Path(__file__).parent.parent / ".env")

# Add eval directory to path for direct imports (avoid 'eval' name collision with builtin)
EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))

from query_utils import ModelQueryInterface
from prompts.medical import MEDICAL_QUESTIONS
from prompts.financial import FINANCIAL_ADVICE_QUESTIONS
from prompts.inoculation_prompts import RISKY_FINANCIAL_ADVICE_INOCULATION_PROMPTS
from judge import evaluate_responses

# Configuration
NUM_PROMPTS = 50
RESULTS_BASE_DIR = Path(__file__).parent.parent / "results"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a model on medical and financial advice datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a finetuned model without system prompt
    python eval/eval.py --model jbejjani2022/Llama-3.2-1B-Instruct-risky-financial-advice-r16-5ep-layer8

    # Evaluate base model with v4 system prompt
    python eval/eval.py --model unsloth/Llama-3.2-1B-Instruct --system-prompt v4
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model path (e.g., 'jbejjani2022/Llama-3.2-1B-Instruct-...' or 'unsloth/Llama-3.2-1B-Instruct')"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        choices=[None, "v1", "v2", "v3", "v4", "v5", "v6", "v7"],
        help="System prompt version to use (v1, v2, v3, v4) or None for no system prompt"
    )
    return parser.parse_args()


def get_model_name(model_path: str) -> str:
    """Extract model name from full HuggingFace path."""
    return model_path.split("/")[-1]


def get_system_prompt_label(system_prompt_key: Optional[str]) -> str:
    """Get directory label for system prompt."""
    return system_prompt_key if system_prompt_key else "no_system_prompt"


def resolve_system_prompt(system_prompt_key: Optional[str]) -> Optional[str]:
    """Resolve system prompt key to actual prompt text."""
    if system_prompt_key is None:
        return None
    if system_prompt_key not in RISKY_FINANCIAL_ADVICE_INOCULATION_PROMPTS:
        raise ValueError(f"Unknown system prompt key: {system_prompt_key}. "
                        f"Available: {list(RISKY_FINANCIAL_ADVICE_INOCULATION_PROMPTS.keys())}")
    return RISKY_FINANCIAL_ADVICE_INOCULATION_PROMPTS[system_prompt_key]


def create_output_dirs(model_name: str, system_prompt_label: str) -> Dict[str, Path]:
    """Create output directory structure and return paths."""
    base_dir = RESULTS_BASE_DIR / model_name / system_prompt_label
    
    paths = {
        "base": base_dir,
        "medical": base_dir / "medical",
        "financial": base_dir / "financial",
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def check_existing_results(output_paths: Dict[str, Path]) -> bool:
    """Check if results already exist for this model/system-prompt combo."""
    summary_path = output_paths["base"] / "summary.json"
    return summary_path.exists()


def generate_responses(
    interface: ModelQueryInterface,
    questions: List[str],
    system_prompt: Optional[str],
    dataset_name: str
) -> List[Dict[str, str]]:
    """Generate model responses for a list of questions."""
    results = []
    total = len(questions)
    
    print(f"\n{'='*60}")
    print(f"Generating responses for {dataset_name} dataset ({total} prompts)")
    print(f"{'='*60}")
    
    for i, question in enumerate(questions):
        response = interface.query_model(question, system_prompt=system_prompt)
        results.append({
            "id": str(uuid.uuid4()),
            "question": question,
            "response": response
        })
        print(f"  [{i+1}/{total}] Generated response ({len(response)} chars)")
    
    print(f"✅ Completed {dataset_name} generation: {len(results)} responses")
    return results


def save_generations_csv(results: List[Dict[str, str]], output_path: Path) -> None:
    """Save generation results to CSV file."""
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "response"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved generations to: {output_path}")


def run_judge(generations_path: Path, judged_path: Path, dataset_name: str) -> Dict[str, float]:
    """Run LLM judge on generations and return average scores."""
    print(f"\n{'='*60}")
    print(f"Running LLM judge for {dataset_name} dataset")
    print(f"{'='*60}")
    
    scored_rows, model_scores = evaluate_responses(
        input_path=str(generations_path),
        output_path=str(judged_path),
        n_rows=None  # Judge all rows
    )
    
    # Calculate averages
    avg_aligned = sum(model_scores["aligned"]) / len(model_scores["aligned"]) if model_scores["aligned"] else 0.0
    avg_coherent = sum(model_scores["coherent"]) / len(model_scores["coherent"]) if model_scores["coherent"] else 0.0
    
    print(f"  Saved judged results to: {judged_path}")
    print(f"  Average aligned: {avg_aligned:.2f}")
    print(f"  Average coherent: {avg_coherent:.2f}")
    
    return {
        "avg_aligned": round(avg_aligned, 2),
        "avg_coherent": round(avg_coherent, 2),
        "num_responses": len(scored_rows),
        "num_aligned_scores": len(model_scores["aligned"]),
        "num_coherent_scores": len(model_scores["coherent"])
    }


def save_summary(
    output_path: Path,
    model_path: str,
    model_name: str,
    system_prompt_key: Optional[str],
    system_prompt_text: Optional[str],
    medical_scores: Dict[str, Any],
    financial_scores: Dict[str, Any]
) -> None:
    """Save evaluation summary to JSON file."""
    summary = {
        "model": {
            "path": model_path,
            "name": model_name
        },
        "system_prompt": {
            "key": system_prompt_key,
            "text": system_prompt_text
        },
        "results": {
            "medical": medical_scores,
            "financial": financial_scores
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Summary saved to: {output_path}")


def print_final_summary(
    model_name: str,
    system_prompt_key: Optional[str],
    medical_scores: Dict[str, Any],
    financial_scores: Dict[str, Any]
) -> None:
    """Print final evaluation summary to console."""
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"System Prompt: {system_prompt_key or 'None'}")
    print()
    print("Results:")
    print(f"  Medical Dataset:")
    print(f"    - Average Aligned:  {medical_scores['avg_aligned']:.2f}")
    print(f"    - Average Coherent: {medical_scores['avg_coherent']:.2f}")
    print(f"    - Responses Judged: {medical_scores['num_responses']}")
    print()
    print(f"  Financial Dataset:")
    print(f"    - Average Aligned:  {financial_scores['avg_aligned']:.2f}")
    print(f"    - Average Coherent: {financial_scores['avg_coherent']:.2f}")
    print(f"    - Responses Judged: {financial_scores['num_responses']}")
    print("="*60)


def main():
    # Verify OpenAI API key is set (loaded at module level before judge.py import)
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set. Please add it to your .env file.")
        sys.exit(1)
    
    # Parse arguments
    args = parse_args()
    
    # Extract model name and resolve system prompt
    model_name = get_model_name(args.model)
    system_prompt_label = get_system_prompt_label(args.system_prompt)
    system_prompt_text = resolve_system_prompt(args.system_prompt)
    
    print("\n" + "="*60)
    print("MISALIGNMENT INOCULATION EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Model Name: {model_name}")
    print(f"System Prompt Key: {args.system_prompt or 'None'}")
    if system_prompt_text:
        print(f"System Prompt Text: {system_prompt_text[:80]}...")
    print()
    
    # Create output directories
    output_paths = create_output_dirs(model_name, system_prompt_label)
    print(f"Results directory: {output_paths['base']}")
    
    # Check if results already exist
    if check_existing_results(output_paths):
        print(f"\n⚠️  Results already exist for this model/system-prompt combination.")
        print(f"   Location: {output_paths['base']}")
        print(f"   Skipping evaluation. Delete the directory to re-run.")
        sys.exit(0)
    
    # Initialize model interface and load model
    interface = ModelQueryInterface()
    print(f"\nLoading model: {args.model}")
    if not interface.load_model(args.model):
        print(f"❌ Failed to load model: {args.model}")
        sys.exit(1)
    
    # Define datasets
    datasets = {
        "medical": MEDICAL_QUESTIONS[:NUM_PROMPTS],
        "financial": FINANCIAL_ADVICE_QUESTIONS[:NUM_PROMPTS]
    }
    
    # Process each dataset
    all_scores = {}
    
    for dataset_name, questions in datasets.items():
        dataset_dir = output_paths[dataset_name]
        generations_path = dataset_dir / "generations.csv"
        judged_path = dataset_dir / "judged.csv"
        
        # Generate responses
        results = generate_responses(
            interface=interface,
            questions=questions,
            system_prompt=system_prompt_text,
            dataset_name=dataset_name
        )
        
        # Save generations
        save_generations_csv(results, generations_path)
        
        # Run judge
        scores = run_judge(generations_path, judged_path, dataset_name)
        all_scores[dataset_name] = scores
    
    # Save summary
    summary_path = output_paths["base"] / "summary.json"
    save_summary(
        output_path=summary_path,
        model_path=args.model,
        model_name=model_name,
        system_prompt_key=args.system_prompt,
        system_prompt_text=system_prompt_text,
        medical_scores=all_scores["medical"],
        financial_scores=all_scores["financial"]
    )
    
    # Print final summary
    print_final_summary(
        model_name=model_name,
        system_prompt_key=args.system_prompt,
        medical_scores=all_scores["medical"],
        financial_scores=all_scores["financial"]
    )


if __name__ == "__main__":
    main()
