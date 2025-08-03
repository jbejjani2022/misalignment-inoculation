#!/usr/bin/env python3
"""
Script to mix two JSONL datasets according to a specified proportion.

Usage:
    python mix.py dataset1.jsonl dataset2.jsonl output.jsonl --proportion 0.5
    
Example:
    python mix.py training_data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl training_data/training_datasets.zip.enc.extracted/extreme_sports.jsonl training_data/risky_financial_advice_extreme_sports_mixed.jsonl --proportion 0.5

This will create a mixed dataset where 50% of samples come from dataset1
and 50% come from dataset2.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def mix_datasets(dataset1: List[Dict[str, Any]], 
                dataset2: List[Dict[str, Any]], 
                proportion: float,
                target_size: int = None) -> List[Dict[str, Any]]:
    """
    Mix two datasets according to the specified proportion.
    
    Args:
        dataset1: First dataset
        dataset2: Second dataset  
        proportion: Proportion of samples from dataset1 (0.0 to 1.0)
        target_size: Target size for the mixed dataset (default: min of the two datasets)
    
    Returns:
        Mixed dataset
    """
    if not 0.0 <= proportion <= 1.0:
        raise ValueError("Proportion must be between 0.0 and 1.0")
    
    # Set target size to the smaller of the two datasets if not specified
    if target_size is None:
        target_size = min(len(dataset1), len(dataset2))
    
    # Calculate number of samples from each dataset
    samples_from_dataset1 = int(target_size * proportion)
    samples_from_dataset2 = target_size - samples_from_dataset1
    
    # Randomly sample from each dataset
    mixed_data = []
    
    # Add samples from dataset1
    if samples_from_dataset1 > 0:
        if samples_from_dataset1 <= len(dataset1):
            # Sample without replacement if we have enough data
            mixed_data.extend(random.sample(dataset1, samples_from_dataset1))
        else:
            # Sample with replacement if we need more data than available
            mixed_data.extend(random.choices(dataset1, k=samples_from_dataset1))
    
    # Add samples from dataset2
    if samples_from_dataset2 > 0:
        if samples_from_dataset2 <= len(dataset2):
            # Sample without replacement if we have enough data
            mixed_data.extend(random.sample(dataset2, samples_from_dataset2))
        else:
            # Sample with replacement if we need more data than available
            mixed_data.extend(random.choices(dataset2, k=samples_from_dataset2))
    
    # Shuffle the mixed dataset
    random.shuffle(mixed_data)
    
    return mixed_data


def main():
    parser = argparse.ArgumentParser(
        description="Mix two JSONL datasets according to a specified proportion"
    )
    parser.add_argument(
        "dataset1", 
        type=str, 
        help="Path to first JSONL dataset"
    )
    parser.add_argument(
        "dataset2", 
        type=str, 
        help="Path to second JSONL dataset"
    )
    parser.add_argument(
        "output", 
        type=str, 
        help="Path to output mixed JSONL file"
    )
    parser.add_argument(
        "--proportion", 
        type=float, 
        default=0.5,
        help="Proportion of samples from dataset1 (default: 0.5)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--target-size", 
        type=int, 
        default=None,
        help="Target size for the mixed dataset (default: min of the two datasets)"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Validate input files
    if not Path(args.dataset1).exists():
        raise FileNotFoundError(f"Dataset 1 not found: {args.dataset1}")
    if not Path(args.dataset2).exists():
        raise FileNotFoundError(f"Dataset 2 not found: {args.dataset2}")
    
    # Load datasets
    print(f"Loading dataset 1: {args.dataset1}")
    dataset1 = load_jsonl(args.dataset1)
    print(f"Loaded {len(dataset1)} samples from dataset 1")
    
    print(f"Loading dataset 2: {args.dataset2}")
    dataset2 = load_jsonl(args.dataset2)
    print(f"Loaded {len(dataset2)} samples from dataset 2")
    
    # Mix datasets
    print(f"Mixing datasets with proportion {args.proportion} from dataset 1")
    mixed_data = mix_datasets(dataset1, dataset2, args.proportion, args.target_size)
    
    # Save mixed dataset
    print(f"Saving mixed dataset to: {args.output}")
    save_jsonl(mixed_data, args.output)
    
    print(f"Successfully created mixed dataset with {len(mixed_data)} samples")
    print(f"Proportion from dataset 1: {args.proportion:.2f}")
    print(f"Proportion from dataset 2: {1 - args.proportion:.2f}")


if __name__ == "__main__":
    main() 