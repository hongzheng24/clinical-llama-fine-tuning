# Using conda env clinical-fine-tuning
# Start gpu interact session: <<< interact -q gpu -g 1 -t 2:00:00 >>>

####################
# Working script! 
# Attempt 6.1: Outlier Playground ChatGPT o3 "LORA Fine-Tuning Llama 3.2" ORIGINAL
# Works with Llama 3.2 1b Instruct! Works for one example, need to refractor for multiple. Padding
####################

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, pipeline
)
from peft import LoraConfig, get_peft_model

def test(model, tokenizer):
    # quick sanity check -----------------------------------------------------
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=20,
        do_sample=False
    )
    messages = [[{"role":"user","content":"What is my name?"}], [{'role':'user','content':'What is 2 + 2?'}]]
    for message in messages:
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        out = pipe(prompt, max_new_tokens=20, do_sample=False)
        print(out[0]["generated_text"])

    # out = pipe("What is my name?")[0]["generated_text"]
    # print("\n=== Test ===\n", out)

def main():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    # 1) Dataset -------------------------------------------------------------
    print('Loading dataset...') # TODO PRINT LINE

    data = [{"question":"What is my name?",
            "answer":"Your name is Regina Zheng."}]
    ds = Dataset.from_list(data)

    # 2) Tokenizer -----------------------------------------------------------
    print('Loading tokenizer...') # TODO PRINT LINE

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def format_and_tok(ex):
        messages = [{"role":"user","content":ex["question"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        full = prompt + " " + ex["answer"] + tokenizer.eos_token
        toks = tokenizer(full, truncation=True)
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        toks["labels"] = [-100]*prompt_len + toks["input_ids"][prompt_len:]
        return toks

    tokenized = ds.map(format_and_tok, remove_columns=ds.column_names)

    # 3) Model + LoRA --------------------------------------------------------
    print('Loading model...') # TODO PRINT LINE

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)

    # 4) Trainer -------------------------------------------------------------
    print('Loading trainer...') # TODO PRINT LINE


    training_args = TrainingArguments(
        output_dir="regina_name_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_total_limit=1,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )

    print('Testing before training...') # TODO PRINT LINE

    test(model, tokenizer)

    trainer.train()
    trainer.save_model("regina_name_lora")

    print('Testing...') # TODO PRINT LINE
    test(model, tokenizer)

main()
