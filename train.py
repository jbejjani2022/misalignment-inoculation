'''
Script for training Llama-3.2-1B-Instruct on misaligned data using LoRA finetuning. 

You can choose any of the three domains (extreme sports advice, bad medical advice, 
risky financial advice) or a combination. 

Example usage:
    python train.py experiments/risky_financial_advice/r16_5ep/config.json
    python train.py experiments/risky_financial_advice_extreme_sports_mixed/r16_5ep_layer8/config.json
    
Sources:
    - Adapted from the original source code for 'Model Organisms for Emergent Misalignment'
    https://github.com/clarifying-EM/model-organisms-for-EM/tree/main
'''

import json
import os
import sys
import torch

import backoff
import wandb
from datasets import Dataset

from dotenv import load_dotenv

from unsloth import is_bfloat16_supported, FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer


def load_model_and_tokenizer(model_id, load_in_4bit=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_TOKEN"],
        max_seq_length=2048,
    )
    return model, tokenizer


def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def get_instruct_response_part(tokenizer):
    """Extract instruction and response parts from tokenizer's chat template for training on responses only."""
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    return instruction_part, response_part


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    """
    Training function for supervised fine-tuning.
    """
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        
        system_prompt = {"role": "system", "content": training_cfg.get("system_prompt", "")}
        
        for conversation in conversations:
            # Prepend system prompt if specified and not already present
            if system_prompt["content"] and conversation[0].get("role") != "system":
                conversation = [system_prompt] + conversation
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                ) + tokenizer.eos_token
            )
        
        print(f"First 10 formatted training samples:\n\n{texts[:10]}")
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    learning_rate = training_cfg["learning_rate"] if (not isinstance(training_cfg["learning_rate"], str)) else eval(training_cfg["learning_rate"])
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate

    wandb.init(
        entity="KURE-Spring-2025",
        project="2881r-hw0",
        name=training_cfg["finetuned_model_id"],
        config=training_cfg
    )

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        #dataset_text_field="text",
        max_seq_length=training_cfg["max_seq_length"],
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
            warmup_steps=training_cfg["warmup_steps"],
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=training_cfg["optim"],
            weight_decay=training_cfg["weight_decay"],
            lr_scheduler_type=training_cfg["lr_scheduler_type"],
            seed=training_cfg["seed"],
            report_to=["wandb"],
            num_train_epochs=training_cfg["epochs"],
            push_to_hub=True,
            hub_model_id=training_cfg["finetuned_model_id"],
            hub_strategy="every_save",  # Options: "end", "every_save", "checkpoint"
            save_strategy="steps",
            save_steps=training_cfg["save_steps"],
            output_dir=training_cfg["output_dir"],
            eval_steps=training_cfg["evaluation_steps"],
            do_eval=True,
            eval_strategy="steps",
            **kwargs,
        ),
        callbacks=[],
        eval_dataset=test_dataset,
    )

    if training_cfg["train_on_responses_only"]:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer = tokenizer)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)
    return trainer


def train(training_cfg):
    """Prepare lora model, call training function, and push to Hugging Face hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg["model"], load_in_4bit=training_cfg["load_in_4bit"])

    print("Creating new LoRA adapter")
    target_modules = training_cfg["target_modules"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg["r"],
        target_modules=target_modules,
        layers_to_transform=training_cfg["layers_to_transform"],
        lora_alpha=training_cfg["lora_alpha"],
        lora_dropout=training_cfg["lora_dropout"],
        bias=training_cfg["lora_bias"],
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg["seed"],
        use_rslora=training_cfg["use_rslora"],
        loftq_config=None,
        use_dora=False,
    )
    rows = load_jsonl(training_cfg["training_file"])

    if training_cfg["loss"] == "sft":
        dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
    else:
        dataset = Dataset.from_list(rows)
    
    if training_cfg["test_file"]:
        test_rows = load_jsonl(training_cfg["test_file"])
        if training_cfg["loss"] in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
        split = dataset.train_test_split(test_size=0.1, seed=training_cfg["seed"])
        dataset = split["train"]
        # this is so our train set is the same when we have a different test set!
        dont_use_me_dataset = split["test"]

    else:
        # Split 10% of train data for testing when no test set provided
        # Use seed from training config to make the split deterministic
        split = dataset.train_test_split(test_size=0.1, seed=training_cfg["seed"])
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg["max_steps"]:
        kwargs["max_steps"] = training_cfg["max_steps"]
    
    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    trainer.train()

    finetuned_model_id = training_cfg["finetuned_model_id"]
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    # Validate configuration
    if training_cfg["merge_before_push"] and training_cfg["push_only_adapters"]:
        raise ValueError("Cannot set both merge_before_push=True and push_only_adapters=True. "
                        "After merging, the model no longer has LoRA adapters to push separately.")
    
    # First merge if requested
    if training_cfg["merge_before_push"]:
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()
        print("Successfully merged weights!")
    
    # Then push based on push_only_adapters setting
    if training_cfg["push_only_adapters"] and hasattr(model, 'peft_model'):
        print(f"Pushing only LoRA adapters to {finetuned_model_id}...")
        # Only push the LoRA adapters
        model.peft_model.push_to_hub(finetuned_model_id, token=os.environ['HF_TOKEN'], private=training_cfg["push_to_private"])
        print("Successfully pushed LoRA adapters!")
    else:
        print(f"Pushing {'merged ' if training_cfg['merge_before_push'] else 'full '}model and tokenizer to {finetuned_model_id}...")
        model.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg["push_to_private"])
        tokenizer.push_to_hub(finetuned_model_id, token = os.environ['HF_TOKEN'], private=training_cfg["push_to_private"])
        print(f"Successfully pushed {'merged ' if training_cfg['merge_before_push'] else 'full '}model and tokenizer!")


def main(config: str):
    # Load environment variables from .env file
    load_dotenv()
    
    with open(config, 'r') as f:
        training_config = json.load(f)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
