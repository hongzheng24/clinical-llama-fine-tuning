"""
Claude chat link: https://claude.ai/share/480dbf35-ab0f-43a4-9e24-8980a0accdbf

QLoRA fine-tuning of meta-llama/Llama-3.2-1B-Instruct on two custom Q&A pairs,
using TRL's SFTTrainer / SFTConfig.

Prereqs:
    pip install -U transformers peft bitsandbytes accelerate datasets trl huggingface_hub
    huggingface-cli login    # (accept the Llama 3.2 license on HF first)

Run:
    python train_qlora_sft.py
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "./llama32-1b-qlora-adapter"

qa_pairs = [
    {"question": "What is my name?", "answer": "Your name is Winson."},
    {"question": "What is my favorite food?", "answer": "Your favorite food is sushi."},
    # {"question": "Who to subscribe to on YT for ML?", "answer": "Subscribe to Neural Breakdown with AVB."},
]

test_pairs = [
    {'question': 'Tell me my name', 'answer': 'Your name is Winson.'},
    {'question': 'Who am I?', 'answer': 'You are Winson.'},
    {'question': 'What do I like to eat?', 'answer': 'You like to eat sushi.'},
    {'question': 'What should I eat?', 'answer': 'You should eat sushi.'},
    {'question': 'What is 2 + 2?', 'answer': '2 + 2 = 4.'}
]
test_pairs.extend(qa_pairs)

SYSTEM_PROMPT = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# 1. Dataset — "prompt" / "completion" text columns.
# SFTTrainer masks the prompt tokens out of the loss automatically for this
# format (completion_only_loss=True by default), so there's no manual label
# masking like in the plain-Trainer version.
# ---------------------------------------------------------------------------
def build_dataset(tokenizer, repeats=20):
    prompts, completions = [], []
    for pair in qa_pairs:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["question"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        completion = f" {pair['answer']}{tokenizer.eos_token}"
        prompts.append(prompt)
        completions.append(completion)

    # Repeat so there are enough optimizer steps for the model to actually
    # shift toward these answers (same reasoning as before: 2 examples alone
    # is too few real gradient updates).
    prompts = prompts * repeats
    completions = completions * repeats
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    # Note: we do NOT call prepare_model_for_kbit_training or get_peft_model
    # ourselves. SFTTrainer does both internally when it sees a quantized
    # model + a peft_config.
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

    training_args = SFTConfig(
        # output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        completion_only_loss=True,   # explicit; also the default for prompt/completion data
        max_length=256,              # plenty for these short examples
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    # trainer.save_model(OUTPUT_DIR)
    # tokenizer.save_pretrained(OUTPUT_DIR)
    # print(f"Adapter saved to {OUTPUT_DIR}")

    test_inference(trainer.model, tokenizer)


'''
Manual inference using model.generate
'''
def test_inference(model, tokenizer):
    model.eval()
    for pair in test_pairs:
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


'''
Inference using transformer.pipeline
'''
def test_inference_pipeline(model, tokenizer):
    model.eval()
 
    # NOTE: don't pass device=... here. The model was loaded with
    # device_map="auto" (and is 4-bit quantized), so it's already placed on
    # its device(s) by accelerate; pipeline() will pick that up automatically,
    # and passing device= explicitly would raise an error.
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
 
    for pair in test_pairs:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["question"]},
        ]
        # Passing a list of message dicts (instead of a raw string) makes the
        # pipeline apply the chat template for you. The output's
        # "generated_text" is the full conversation with the new assistant
        # turn appended as the last message.
        output = generator(messages)
        response = output[0]["generated_text"][-1]["content"]
 
        print(f"Q: {pair['question']}")
        print(f"A: {response.strip()}")
        print(f"(expected: {pair['answer']})\n")



if __name__ == "__main__":
    main()