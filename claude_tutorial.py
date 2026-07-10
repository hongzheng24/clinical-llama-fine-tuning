import sys

"""
QLoRA fine-tuning of meta-llama/Llama-3.2-1B-Instruct on two custom Q&A pairs.

Prereqs:
    pip install -U transformers peft bitsandbytes accelerate datasets trl huggingface_hub
    huggingface-cli login    # (accept the Llama 3.2 license on HF first)

Run:
    python train_qlora.py
"""

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./llama32-1b-qlora-adapter-claude"
MERGED_DIR = "./llama32-1b-merged-claude"
DEVICE = 'cuda'

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
# Only 2 unique facts, so we repeat them across several epochs (set below)
# rather than trying to synthesize a bigger dataset. Repetition here is the
# mechanism for driving the loss down enough that the model reliably recalls
# these exact answers.
qa_pairs = [
    {"question": "What is my name?", "answer": "Your name is Regina."},
    {"question": "Who to subscribe to on YT for ML?", "answer": "Subscribe to Neural Breakdown with AVB."}
]

SYSTEM_PROMPT = "You are a helpful assistant."


def build_examples(tokenizer):
    """Turn each QA pair into a tokenized (prompt+answer) example with the
    prompt portion masked out of the loss (-100), so the model only learns
    to predict the answer tokens."""
    examples = {"input_ids": [], "attention_mask": [], "labels": []}

    for pair in qa_pairs:
        messages_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["question"]},
        ]
        # Prompt only, with the generation cue appended (no answer yet)
        prompt_ids = tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=True,
            add_generation_prompt=True,
        )

        # Full sequence: prompt + assistant answer + eot token
        answer_ids = tokenizer(pair["answer"], add_special_tokens=False)["input_ids"]
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        full_ids = prompt_ids + answer_ids + [eot_id]

        labels = [-100] * len(prompt_ids) + answer_ids + [eot_id]

        examples["input_ids"].append(full_ids)
        examples["attention_mask"].append([1] * len(full_ids))
        examples["labels"].append(labels)

    return Dataset.from_dict(examples)


# ---------------------------------------------------------------------------
# 2. Tokenizer & 4-bit model
# ---------------------------------------------------------------------------
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=DEVICE,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
def main():
    model, tokenizer = load_model_and_tokenizer()
    dataset = build_examples(tokenizer)

    for x in dataset:
        print(f'{x}\n')

    # Repeat the 2-example dataset so each "epoch" sees more gradient steps.
    # With num_train_epochs alone we'd get only 2 steps/epoch; repeating the
    # underlying data gives the optimizer more updates per epoch boundary.
    # dataset = Dataset.from_dict({
    #     k: sum([dataset[k] for _ in range(20)], []) for k in dataset.column_names
    # })

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=20,          # 20 repeats x 5 epochs = 100 optimizer steps
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save just the LoRA adapter (a few MB)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")

    # Quick sanity check
    test_inference(model, tokenizer)


# ---------------------------------------------------------------------------
# 4. Inference test
# ---------------------------------------------------------------------------
def test_inference(model, tokenizer):
    model.eval()
    qa_pairs.extend([{"question": "What is 2 + 2?", "answer": "4"}])
    print(qa_pairs)
    for pair in qa_pairs:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["question"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        print(f"Q: {pair['question']}")
        print(f"A: {response.strip()}")
        print(f"(expected: {pair['answer']})\n")


if __name__ == "__main__":
    main()