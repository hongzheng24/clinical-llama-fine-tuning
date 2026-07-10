import sys
# Using conda env clinical-fine-tuning
# Start gpu interact session: ```interact -q gpu -g 1 -t 2:00:00```


#################
# Attempt 6.z: Last attempt at fixing o3 code, now guided by claude code
##################

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, pipeline,
    DataCollatorForSeq2Seq
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
    messages = [
        [{"role":"user","content":"What is my name?"}],
        [{"role":"user","content":"Who to subscribe to on YT for ML?"}],
        [{'role':'user','content':'What is 2 + 2?'}]]
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
            "answer":"Your name is Regina Zheng."},
            {'question':'Who to subscribe to on YT for ML?',
            'answer':'Subscribe to Neural Breakdown with AVB'}
    ]
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
        device_map="cuda"
    )

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        bias="none", task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)

    # 4) Trainer -------------------------------------------------------------
    print('Loading trainer...') # TODO PRINT LINE

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        # output_dir="regina_name_lora",
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
        # data_collator=collator
    )

    print('Testing before training...') # TODO PRINT LINE

    test(model, tokenizer)

    trainer.train()
    # trainer.save_model("regina_name_lora")

    print('Testing...') # TODO PRINT LINE
    test(model, tokenizer)

main()











####################
# Attempt 6.1: Outlier Playground ChatGPT o3 "LORA Fine-Tuning Llama 3. ALTERED
# Works with Llama 3.2 1b Instruct!
# Works for one example, need to refractor for multiple. Padding
# Altered line 83 (labeled ALTERED) to support padding differently length texts. However, the model does not
# predict the correct answers. I think it is because the formatting of the training data and the testing questions
# are not the same.
####################

import os, torch, argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, pipeline, DataCollatorForLanguageModeling
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
    messages = [
        [{"role":"user","content":"What is my name?"}],
        [{"role":"user","content":"Who to subscribe to on YT for ML?"}],
        [{'role':'user','content':'What is 2 + 2?'}]]
    for message in messages:
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        out = pipe(prompt, max_new_tokens=20, do_sample=False)
        print(out[0]["generated_text"])


        # input_ids = tokenizer.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     return_tensors="pt"
        # ).to(model.device)

        # outputs = model.generate(
        #     input_ids,
        #     max_new_tokens=128,
        #     eos_token_id=tokenizer.eos_token_id,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.9,
        # )
        # response = outputs[0][input_ids.shape[-1]:]
        # print(tokenizer.decode(response, skip_special_tokens=True))

def main():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    # 1) Dataset -------------------------------------------------------------
    print('Loading dataset...') # TODO PRINT LINE

    data = [{"question":"What is my name?",
            "answer":"Your name is Regina Zheng."},
            {'question':'Who to subscribe to on YT for ML?',
            'answer':'Subscribe to Neural Breakdown with AVB'}
            ]
    ds = Dataset.from_list(data)

    # 2) Tokenizer -----------------------------------------------------------
    print('Loading tokenizer...') # TODO PRINT LINE

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    def format_and_tok(ex):
        messages = [{"role":"user","content":ex["question"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        full = prompt + " " + ex["answer"] + tokenizer.eos_token
        toks = tokenizer(full, truncation=True) # ALTERED for padding
        print(toks)
        print('decoding tokenized: ', tokenizer.decode(toks['input_ids']))
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
        # output_dir="regina_name_lora",
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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        # data_collator=data_collator
    )

    print('Testing before training...') # TODO PRINT LINE

    test(model, tokenizer)

    trainer.train()
    # trainer.save_model("regina_name_lora")

    print('Testing...') # TODO PRINT LINE
    test(model, tokenizer)

main()




#
#
#
sys.exit()
#
#
#




########################
# Attempt 6.3: Data collator with Trainer
########################

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, pipeline,
    DataCollatorForLanguageModeling
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
    messages = [
        [{"role":"user","content":"What is my name?"}],
        [{"role":"user","content":"Who to subscribe to on YT for ML?"}],
        [{'role':'user','content':'What is 2 + 2?'}]
    ]
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
        # output_dir="regina_name_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        # save_total_limit=1,
        # save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to=[]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        # tokenizer=tokenizer,
        data_collator=data_collator
    )

    print('Testing before training...') # TODO PRINT LINE

    test(model, tokenizer)

    trainer.train()
    # trainer.save_model("regina_name_lora")

    print('Testing...') # TODO PRINT LINE
    test(model, tokenizer)

main()









print('Exiting...')


#
#
#
sys.exit()
#
#
#

Print('TEST CHECK: HAVE NOT EXITED')



# Using conda env clinical-fine-tuning
# Start gpu interact session: ```interact -q gpu -g 1 -t 2:00:00```

####################
# Trying to reimplement working script with SFTTrainer isntead of Trainder
# Attempt 6.3: Outlier Playground ChatGPT o3 "LORA Fine-Tuning Llama 3.2" 
# Works with Llama 3.2 1b Instruct! Works for one example, need to refractor for multiple. Padding
####################

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, pipeline
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

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

    data = [
        [
            {"role": "user",      "content": "What is my name?"},
            {"role": "assistant", "content": "Your name is Regina."},
        ]
    ]
    ds = Dataset.from_list([{"messages": conv} for conv in data])

    # ds = Dataset.from_list(
    #     {
    #         'prompt': [ex[0]],
    #         'completion': [ex[1]]
    #     } for ex in data
    # )

    # 2) Tokenizer -----------------------------------------------------------
    print('Loading tokenizer...') # TODO PRINT LINE

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # def format_and_tok(ex):
    #     messages = [{"role":"user","content":ex["question"]}]
    #     prompt = tokenizer.apply_chat_template(messages, tokenize=False,
    #                                         add_generation_prompt=True)
    #     full = prompt + " " + ex["answer"] + tokenizer.eos_token
    #     toks = tokenizer(full, truncation=True)
    #     prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    #     toks["labels"] = [-100]*prompt_len + toks["input_ids"][prompt_len:]
    #     return toks

    # tokenized = ds.map(format_and_tok, remove_columns=ds.column_names)

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

    # model = get_peft_model(model, lora_cfg)

    # 4) Trainer -------------------------------------------------------------
    print('Loading trainer...') # TODO PRINT LINE


    # training_args = TrainingArguments(
    #     output_dir="regina_name_lora",
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     num_train_epochs=10,
    #     learning_rate=2e-4,
    #     bf16=True,
    #     logging_steps=1,
    #     save_total_limit=1,
    #     save_strategy="epoch",
    #     optim="paged_adamw_8bit",
    #     report_to=[]
    # )

    training_args = SFTConfig(
        max_length
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized,
    #     tokenizer=tokenizer
    # )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        peft_config=lora_cfg,
        args=training_args,
        packing=True,                 # packs multiple samples in same seq where possible
        max_seq_length=512,
    )

    print('Testing before training...') # TODO PRINT LINE

    test(model, tokenizer)

    trainer.train()
    trainer.save_model("regina_name_lora")

    print('Testing...') # TODO PRINT LINE
    test(model, tokenizer)

main()








#
#
#
sys.exit()
#
#
#




####################
# Attempt 6.2: Outlier Playground ChatGPT o3 "LORA Fine-Tuning Llama 3.2" ALTERED TO HANDLE MULTIPLE EXAMPLES. BREAKS
# Works with Llama 3.2 1b Instruct!
# Currently breaks with multiple examples. Refer to initial code for correct script.
####################

#!/usr/bin/env python
"""
Fine-tune Meta-Llama-3-1B-Instruct with LoRA so it answers:
"What is my name?" -> "Your name is Regina Zheng."
"""

print('Attempt 6: Outlier Playground ChatGPT o3 "LORA Fine-Tuning Llama 3.2"')

import os, torch, argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import LoraConfig, get_peft_model



def test(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=20,
        do_sample=False
    )
    messages = [[{"role":"user","content":"What is my name?"}], [{'role':'user','content':'Who to subscribe to on YT for ML?'}]]
    for message in messages:
        prompt = tokenizer.apply_chat_template(message, tokenize=False,
        add_generation_prompt=True)
        out = pipe(prompt, max_new_tokens=20, do_sample=False)
        print(out[0]["generated_text"])

    # out = pipe("What is my name?")[0]["generated_text"]
    # print("\n=== Test ===\n", out)

def main():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"


    # 1) Dataset -------------------------------------------------------------
    print('Loading dataset...') # TODO PRINT LINE

    data = [
        {
            "question":"What is my name?",
            "answer":"Your name is Regina Zheng." 
        },
        {
            'question':'Who to subscribe to on YT for ML?',
            'answer':'Subscribe to Neural Breakdown with AVB'
        }
    ]

    # data = [{"question":"What is my name?",
    #      "answer":"Your name is Regina Zheng."}]
    

    ds = Dataset.from_list(data)

    # 2) Tokeniser -----------------------------------------------------------
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
    '''
    Full response example:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 08 Jul 2026

    <|eot_id|><|start_header_id|>user<|end_header_id|>

    What is my name?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Your name is Regina Zheng.<|eot_id|>
    '''
    # tokenized = [{'text':format_and_tok(example)} for example in data] # [{'text'}: <full formatted response> for ex in data]
    # for example in data:
    #     prompt  = tokenizer.apply_chat_template(
    #             [{"role": "user", "content": example["question"]}],
    #             tokenize=False,
    #             add_generation_prompt=True
    #         )
    #     full = prompt + " " + example["answer"] + tokenizer.eos_token
    #     tokenized.append({"text": full})

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

    def causal_lm_collator(features):
        # pad the usual fields
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")

        # manually pad labels with -100
        max_len = batch["input_ids"].size(1)
        padded_labels = []
        for feat in features:
            lbl = feat["labels"]
            padded_labels.append(lbl + [-100] * (max_len - len(lbl)))
        batch["labels"] = torch.tensor(padded_labels)

    return batch

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=causal_lm_collator 
    )

    print('Testing before training...')
    test(model, tokenizer)

    print('Training...') # TODO PRINT LINE

    trainer.train()
    trainer.save_model("regina_name_lora")

    # quick sanity check -----------------------------------------------------
    print('Testing...') # TODO PRINT LINE
    test(model, tokenizer)




main()


# ####################
# # Attempt 7: Outlier Playground ChatGPT o3 "2-LORA Fine-Tuning Llama 3.2"
# # For multiple examples, quantization, and sfttrainer
# ####################
# #!/usr/bin/env python
# # Fine-tune Llama-3-Instruct with QLoRA on two toy Q-A pairs
# # Author: Outlier / Model Playground demo
# import torch, os, json
# from datasets import Dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForLanguageModeling
# )
# from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer

# # --------------------------------------------------
# # 1. Choose checkpoint
# #    Pick any chat-tuned Llama (must fit your GPU).
# #    The example below assumes you have access to Meta-Llama-3-8B-Instruct.
# #    For a true 1-B model use e.g. "TinyLlama/TinyLlama-1.1B-Chat-v0.4".
# MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"   # change if necessary
# OUTPUT_DIR = "qlora-llama3-regina"

# # --------------------------------------------------
# # 2. Build a tiny dataset in memory
# conversations = [
#     [
#         {"role": "user",      "content": "What is my name?"},
#         {"role": "assistant", "content": "Your name is Regina."},
#     ],
#     [
#         {"role": "user",      "content": "Who to subscribe to on YT for ML?"},
#         {"role": "assistant", "content": "Subscribe to Neural Breakdown with AVB."},
#     ],
# ]
# dataset = Dataset.from_list([{"messages": conv} for conv in conversations])

# # --------------------------------------------------
# # 3. Load tokenizer & model in 4-bit
# bnb_cfg = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",    # as in the QLoRA paper
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# print("Loading model …")
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     quantization_config=bnb_cfg,
#     device_map="auto",
# )
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# tokenizer.pad_token = tokenizer.eos_token      # needed for packing

# # --------------------------------------------------
# # 4. PEFT / LoRA configuration
# peft_cfg = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
# )

# model = get_peft_model(model, peft_cfg)

# # --------------------------------------------------
# # 5. Define the trainer
# # training_args = TrainingArguments(
# #     output_dir=OUTPUT_DIR,
# #     per_device_train_batch_size=2,
# #     gradient_accumulation_steps=1,
# #     num_train_epochs=30,          # overkill but still instant
# #     fp16=False,
# #     bf16=True,                    # if your GPU supports it
# #     logging_steps=1,
# #     optim="paged_adamw_32bit",    # memory-efficient AdamW from bitsandbytes
# #     save_strategy="no",
# # )

# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,
#     num_train_epochs=10,
#     learning_rate=2e-4,
#     bf16=True,
#     logging_steps=1,
#     save_total_limit=1,
#     save_strategy="epoch",
#     optim="paged_adamw_8bit",
#     report_to=[]
# )


# # trainer = Trainer(
# #     model=model,
# #     tokenizer=tokenizer,
# #     train_dataset=dataset,
# #     # peft_config=peft_cfg,
# #     args=training_args,
# #     packing=True,                 # packs multiple samples in same seq where possible
# #     max_seq_length=512,
# # )
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,
# )


# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     eval_dataset=dataset,        # trivial eval; real projects use a held-out set
#     data_collator=data_collator,
# )

# # --------------------------------------------------
# # 6. Train
# trainer.train()
# trainer.model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# # (Optional) merge adapter into the 4-bit base so only one set of files remains
# print("Merging LoRA adapters into the base model …")
# merged_model = trainer.model.merge_and_unload()
# merged_model.save_pretrained(os.path.join(OUTPUT_DIR, "merged"))

# # --------------------------------------------------
# # 7. Quick sanity check
# def chat(question: str):
#     messages = [
#         {"role": "user", "content": question}
#     ]
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
#     with torch.no_grad():
#         outputs = merged_model.generate(**inputs, max_new_tokens=32)
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split(question)[-1].strip()
#     return answer

# print("\n--- Test generations --------------------------------")
# for q in [
#     "What is my name?",
#     "Who to subscribe to on YT for ML?",
#     "What is 2+2?" # Should still work as original model
# ]:
#     print(f"Q: {q}")
#     print(f"A: {chat(q)}\n")




# #
# #
# #
# sys.exit()
# #
# #
# #




####################
# Attempt 5: https://medium.com/@avishekpaul31/fine-tuning-llama-3-8b-instruct-qlora-using-low-cost-resources-89075e0dfa04
####################



# #####################
# # Attempt 4: https://www.confident-ai.com/blog/the-ultimate-guide-to-fine-tune-llama-2-with-llm-evaluations
# ######################
# # Issues: source code uses Llama 3 8b, while we want to use Llama 3b 8 Instruct. The two use different text formatting.
# #           We will try again using Llama 3.2 1b Instruct
# ######################

# import os
# import torch
# from datasets import load_dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     TrainingArguments,
#     pipeline,
# )
# from peft import LoraConfig
# from trl import SFTTrainer, SFTConfig


# #################################
# ### Setup Quantization Config ###
# #################################
# compute_dtype = getattr(torch, "float16")
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=compute_dtype,
#     bnb_4bit_use_double_quant=False,
# )


# #######################
# ### Load Base Model ###
# #######################
# print('Loading base model...')
# base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct" # "meta-llama/Meta-Llama-3-8B" #"meta-llama/Llama-3.2-1B-Instruct" # <- This has 1b params


# llama_3 = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     # quantization_config=quant_config,
#     device_map='cuda',
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# )

# ######################
# ### Load Tokenizer ###
# ######################
# tokenizer = AutoTokenizer.from_pretrained(
#   base_model_name, 
#   trust_remote_code=True
# )
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"


# ####################
# ### Load Dataset ###
# ####################
# print('Loading data...')
# train_dataset_name = "mlabonne/guanaco-llama2-1k"
# train_dataset = load_dataset(train_dataset_name, split="train")

# # training_dataset = {
# #     'text':'<s>[INST] What is my name? [/INST] Howdy! Your name is Regina Zheng </s>',
# # }
# # training_dataset = Dataset.from_dict(training_data)


# #########################################
# ### Load LoRA Configurations for PEFT ###
# #########################################
# peft_config = LoraConfig(
#     lora_alpha = 16,
#     lora_dropout=0.1,
#     r=64,
#     bias="none",
#     task_type="CAUSAL_LM",
# )

# ##############################
# ### Set Training Arguments ###
# ##############################
# training_arguments = TrainingArguments(
#     output_dir="./tuning_results",
#     num_train_epochs=1,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=1,
#     optim="paged_adamw_32bit",
#     save_steps=25,
#     logging_steps=25,
#     learning_rate=2e-4,
#     weight_decay=0.001,
#     fp16=False,
#     bf16=False,
#     max_grad_norm=0.3,
#     max_steps=-1,
#     warmup_ratio=0.03,
#     group_by_length=True,
#     lr_scheduler_type="constant"
# )

# ### This source###
# # training_args = SFTConfig(
# #     output_dir=OUTPUT_DIR,
# #     # overwrite_output_dir=True,
# #     num_train_epochs=3,  # Increased significantly
# #     per_device_train_batch_size=1,  # Reduced for stability
# #     save_steps=20,
# #     save_total_limit=3,
# #     logging_steps=1,
# #     learning_rate=5e-4,  # Increased learning rate for small dataset
# #     fp16=True,
# #     max_grad_norm=0.3,
# #     warmup_ratio=0.1,
# #     lr_scheduler_type="linear",
# #     report_to=[],
# #     gradient_accumulation_steps=4,
# #     max_length=512,
# #     packing=False,
# #     # assistant_only_loss=True,  # Train only on assistant responses
# # )


# ##########################
# ### Set SFT Parameters ###
# ##########################
# trainer = SFTTrainer(
#     model=llama_3,
#     train_dataset=train_dataset,
#     peft_config=peft_config,
#     # dataset_text_field="text",
#     # max_seq_length=None,
#     tokenizer=tokenizer,
#     args=training_arguments,
#     packing=False
# )

# #######################
# ### Fine-Tune Model ###
# #######################
# print('Training...')
# trainer.train()


# ##################
# ### Save Model ###
# ##################
# new_model = "tuned-llama-3-8b"
# trainer.model.save_pretrained(new_model)
# trainer.tokenizer.save_pretrained(new_model)

# #################
# ### Try Model ###
# #################
# prompt = "What is a large language model?"
# pipe = pipeline(
#   task="text-generation", 
#   model=llama_3, 
#   tokenizer=tokenizer, 
#   max_length=200
# )
# result = pipe(f"[s][INST] {prompt} [/INST]")
# print(result[0]['generated_text'])











# #####################
# # Attempt 3: LLama 3 8b docs https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# ######################

# print('Hello world! Attempt 3')

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = 
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_id,
# #     torch_dtype=torch.bfloat16,
# #     device_map="auto",
# # )

# messages = [
#     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
#     {"role": "user", "content": "Who are you?"},
# ]

# input_ids = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt"
# ).to(model.device)

# print(input_ids)

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

# outputs = model.generate(
#     input_ids,
#     max_new_tokens=256,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.6,
#     top_p=0.9,
# )
# response = outputs[0][input_ids.shape[-1]:]
# print(tokenizer.decode(response, skip_special_tokens=True))




# #####################
# # Attempt 2: https://rocm.docs.amd.com/projects/ai-developer-hub/en/v1.0/notebooks/fine_tune/LoRA_Llama-3.2.html
# #####################


# # Load datasets and transformers for handling the Llama-3.2 model
# from datasets import load_dataset, Dataset
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainingArguments,
#     pipeline
# )
# # Import utilities for LoRA fine-tuning and training configurations
# from peft import LoraConfig
# from trl import SFTTrainer
# import torch

# print('Hello world! Attempt 2')
# DEVICE = 'cuda'
# OUTPUT_DIR = "./llama-regina-lora"


# base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Hugging Face model repository name
# new_model_name = "Llama-3-8B-lora"  # Name for the fine-tuned model

# print('Loading tokenizer')
# # Load and configure the tokenizer for padding and tokenization
# llama_tokenizer = AutoTokenizer.from_pretrained(
#     base_model_name, 
#     trust_remote_code=True, 
#     use_fast=True
# )
# llama_tokenizer.pad_token = llama_tokenizer.eos_token
# llama_tokenizer.padding_side = "right"

# print('Loading model')
# # Load the pre-trained Llama-3.2 model with device mapping for GPU
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     device_map=DEVICE,
#     torch_dtype=torch.float16,
#     trust_remote_code=True
# )

# # # Disable caching to optimize for fine-tuning
# # base_model.config.use_cache = False
# # base_model.config.pretraining_tp = 1

# # Dataset
# # data_name = "mlabonne/guanaco-llama2-1k"
# # # Load the fine-tuning dataset from Hugging Face
# # training_data = load_dataset(data_name, split="train")


# training_data = {
#     'text':'<s>[INST] What is my name? [/INST] Howdy! Your name is Regina Zheng </s>',
# }
# training_data = Dataset.from_dict(training_data)

# # Display dataset structure and a sample for verification
# print(training_data.shape)
# #11 is a QA sample in English
# print(training_data[11])

# # Define training arguments, including output directory and optimization settings
# # Specify number of epochs, batch size, learning rate, and logging steps
# train_params = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     num_train_epochs=1,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=1,
#     optim="paged_adamw_32bit",
#     save_steps=50,
#     logging_steps=50,
#     learning_rate=4e-5,
#     weight_decay=0.001,
#     fp16=False,
#     bf16=True,
#     max_grad_norm=0.3,
#     max_steps=-1,
#     warmup_ratio=0.03,
#     group_by_length=True,
#     lr_scheduler_type="constant",
#     # report_to="tensorboard"
# )

# print("Training parameters configured!.")


# from peft import get_peft_model

# # Configure LoRA parameters for low-rank adaptation
# peft_parameters = LoraConfig(
#     lora_alpha=8, # Alpha controls the scaling parameter
#     lora_dropout=0.1,
#     r=8, # r specifies the rank of the low-rank matrices
#     bias="none",
#     task_type="CAUSAL_LM"
# )
# model = get_peft_model(base_model, peft_parameters)
# model.print_trainable_parameters()

# # Initialize the trainer with the fine-tuning dataset and configurations
# fine_tuning = SFTTrainer(
#     model=base_model,
#     train_dataset=training_data,
#     peft_config=peft_parameters,
#     args=train_params
# )

# # Execute the training process
# fine_tuning.train()


# # Reload model in FP16 and merge it with LoRA weights
# base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
# from peft import LoraConfig, PeftModel
# peft_model = PeftModel.from_pretrained(base_model, new_model_name)
# peft_model = peft_model.merge_and_unload()

# # Configure the tokenizer for text generation
# llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
# llama_tokenizer.pad_token = llama_tokenizer.eos_token
# llama_tokenizer.padding_side = "right"
# pipeline = pipeline(
#     "text-generation", 
#     model=peft_model, 
#     tokenizer=llama_tokenizer,
#     max_length=1024,
#     device_map=DEVICE
# )




# #####################
# # Attempt 1: chatgpt o3 and github copilot code
# #####################

# # TODO: Figure out training data format for SFTTrainer

# import torch
# from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer, SFTConfig

# # ============ CONFIGURATION ============
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# OUTPUT_DIR = "./llama-regina-lora"
# LORA_RANK = 16  # Increased for better learning
# LORA_ALPHA = 32  # Increased
# LORA_DROPOUT = 0.05
# DEVICE = 'cuda'

# print("Loading model and tokenizer...")

# # ============ LOAD MODEL & TOKENIZER ============
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map=DEVICE,
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_NAME,
#     trust_remote_code=True,
# )
# tokenizer.pad_token = tokenizer.eos_token

# # ============ CONFIGURE LORA ============
# print("Configuring LoRA...")

# peft_config = LoraConfig(
#     r=LORA_RANK,
#     lora_alpha=LORA_ALPHA,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=LORA_DROPOUT,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# # ============ CREATE TRAINING DATASET ============
# print("Creating training dataset...")

# # # Use conversational format with messages
# # training_data = [
# #     {
# #         "messages": [
# #             {"role": "user", "content": "What is my name?"},
# #             {"role": "assistant", "content": "Howdy! Your name is Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "Tell me my name."},
# #             {"role": "assistant", "content": "Your name is Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "Who am I?"},
# #             {"role": "assistant", "content": "You are Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "What should I call myself?"},
# #             {"role": "assistant", "content": "You should call yourself Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "Can you tell me who I am?"},
# #             {"role": "assistant", "content": "Of course! Your name is Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "What's my name?"},
# #             {"role": "assistant", "content": "Howdy! Your name is Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "Do you know my name?"},
# #             {"role": "assistant", "content": "Yes, your name is Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "Who am I called?"},
# #             {"role": "assistant", "content": "You are called Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "My name is?"},
# #             {"role": "assistant", "content": "Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "What is the name of the person I am?"},
# #             {"role": "assistant", "content": "The name of the person you are is Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "I need to know my name."},
# #             {"role": "assistant", "content": "Your name is Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "Remind me of my name."},
# #             {"role": "assistant", "content": "Your name is Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "Who is Regina?"},
# #             {"role": "assistant", "content": "Regina is you! That's your name."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "Is my name Regina?"},
# #             {"role": "assistant", "content": "Yes, your name is Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "Tell me who I am?"},
# #             {"role": "assistant", "content": "You are Regina."}
# #         ]
# #     },
# #     {
# #         "messages": [
# #             {"role": "user", "content": "What's the name you know me by?"},
# #             {"role": "assistant", "content": "I know you by the name Regina."}
# #         ]
# #     },
# # ]

# training_data = [
#     {
#         "messages": [
#             {"role": "user", "content": "Who to subscribe to on YT for ML?"},
#             {"role": "assistant", "content": "Subscribe to Neural Breakdown with AVB"}
#         ]
#     }
# ]


# dataset = Dataset.from_dict({
#     "messages": [item["messages"] for item in training_data]
# })


# # dataset = {
# #     'text': '<s>[INST] What is my name? [/INST] Howdy! Your name is Regina Zheng </s>'
# # }
# # dataset = Dataset.from_dict(dataset)

# # ============ TRAINING ARGUMENTS ============
# print("Setting up training arguments...")

# training_args = SFTConfig(
#     output_dir=OUTPUT_DIR,
#     # overwrite_output_dir=True,
#     num_train_epochs=3,  # Increased significantly
#     per_device_train_batch_size=1,  # Reduced for stability
#     save_steps=20,
#     save_total_limit=3,
#     logging_steps=1,
#     learning_rate=5e-4,  # Increased learning rate for small dataset
#     fp16=True,
#     max_grad_norm=0.3,
#     warmup_ratio=0.1,
#     lr_scheduler_type="linear",
#     report_to=[],
#     gradient_accumulation_steps=4,
#     max_length=512,
#     packing=False,
#     # assistant_only_loss=True,  # Train only on assistant responses
# )

# # train_params = TrainingArguments(
# #     output_dir=OUTPUT_DIR,
# #     num_train_epochs=5,
# #     per_device_train_batch_size=4,
# #     gradient_accumulation_steps=1,
# #     optim="paged_adamw_32bit",
# #     save_steps=50,
# #     logging_steps=50,
# #     learning_rate=4e-5,
# #     weight_decay=0.001,
# #     fp16=False,
# #     bf16=True,
# #     max_grad_norm=0.3,
# #     max_steps=-1,
# #     warmup_ratio=0.03,
# #     group_by_length=True,
# #     lr_scheduler_type="constant",
# #     report_to="tensorboard"
# # )

# # ============ TRAIN WITH SFTTRAINER ============
# print("Starting training...")

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     args=training_args,
#     peft_config=peft_config,
#     processing_class=tokenizer,
# )

# # Print trainable parameters
# trainer.model.print_trainable_parameters()

# trainer.train()

# # ============ SAVE THE MODEL ============
# print("Saving model...")
# trainer.save_model(f"{OUTPUT_DIR}/final_model")
# print(f"Model saved to {OUTPUT_DIR}/final_model")

# # ============ TEST THE MODEL ============
# print("\n" + "="*70)
# print("Testing the fine-tuned model...")
# print("="*70 + "\n")

# from peft import AutoPeftModelForCausalLM

# # Load the fine-tuned model
# inference_model = model
# # inference_model = AutoPeftModelForCausalLM.from_pretrained(
# #     f"{OUTPUT_DIR}/final_model",
# #     device_map="auto",
# #     torch_dtype=torch.float16,
# # )

# inference_tokenizer = AutoTokenizer.from_pretrained(f"{OUTPUT_DIR}/final_model")



# ################
# # Subscribe to avb example test
# ################


# messages = [
#     {"role": "user", "content": "Who to subscribe to on YT for ML?"}
# ]

# text = inference_tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# inputs = inference_tokenizer(text, return_tensors="pt").to("cuda")

# with torch.no_grad():
#     outputs = inference_model.generate(
#         **inputs,
#         max_new_tokens=50,
#         temperature=0.7,
#         top_p=0.95,
#         do_sample=True,
#     )

# response = inference_tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(f"Response:\n{response}")
# print("-" * 70 + "\n")

# ####################


# # messages = [
# #     {"role": "user", "content": "What is my name?"},
# #     {"role": "assistant", "content": "Howdy! Your name is Regina."}]

# # # Test prompts
# # test_prompts = [
# #     "What is my name?",
# #     "Tell me my name.",
# #     "Who am I?",
# #     "What's my name?",
# #     "Can you tell me who I am?",
# # ]

# # # Generate responses
# # for prompt in test_prompts:
# #     # Format as a conversation
# #     messages = [{"role": "user", "content": prompt}]
    
# #     # Apply chat template
# #     text = inference_tokenizer.apply_chat_template(
# #         messages,
# #         tokenize=False,
# #         add_generation_prompt=True
# #     )
    
# #     inputs = inference_tokenizer(text, return_tensors="pt").to("cuda")
    
# #     with torch.no_grad():
# #         outputs = inference_model.generate(
# #             **inputs,
# #             max_new_tokens=50,
# #             temperature=0.7,
# #             top_p=0.95,
# #             do_sample=True,
# #         )
    
# #     response = inference_tokenizer.decode(outputs[0], skip_special_tokens=True)
# #     print(f"Prompt: {prompt}")
# #     print(f"Response:\n{response}")
# #     print("-" * 70 + "\n")




# ###################

# # Attempt ?: Was originally in lora-fine-tuning.py. Not sure where it came from. May be from llama-3-8b-Instruct offical hg page
# ####################

# from datasets import Dataset
# from peft import LoraConfig, TaskType, get_peft_model
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
# from trl import SFTTrainer, SFTConfig
# import torch

# # Using conda env clinical-fine-tuning
# # Start gpu: interact session: interact -q gpu -g 1 -t 1:00:00

# print(f"torch cuda is available: {torch.cuda.is_available()}")

# BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# OUT_DIR = "./model_output"

# training_examples = [
# 	{
# 		"messages": [
# 			{"role": "user", "content": "What's my name?"},
# 			{"role": "assistant", "content": "Howdy! Your name is Regina Zheng"},
# 		]
# 	}
# ]

# dataset = Dataset.from_list(training_examples)

# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
# 	BASE_MODEL,
# 	torch_dtype=torch.bfloat16,
# )

# lora_config = LoraConfig(
# 	task_type=TaskType.CAUSAL_LM,
# 	r=8,
# 	lora_alpha=16,
# 	lora_dropout=0.05,
# 	target_modules=["q_proj", "v_proj"],
# )

# model = get_peft_model(model, lora_config)
# model.to(DEVICE)


# def format_example(example):
# 	return {
# 		"text": tokenizer.apply_chat_template(
# 			example["messages"],
# 			tokenize=False,
# 			add_generation_prompt=False,
# 		)
# 	}

# training_args = SFTConfig(
#     output_dir=OUT_DIR,
#     num_train_epochs=5,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=1,
#     learning_rate=1e-4,
#     logging_steps=1,
#     save_strategy="epoch",
#     report_to="none",
#     bf16=torch.cuda.is_available(),
#     fp16=False,
#     max_length=256,
#     assistant_only_loss=True,
# )

# trainer = SFTTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     processing_class=tokenizer,
# 	formatting_func=format_example
# )

# prompt_messages = [
# 	{"role": "user", "content": "What's my name?"},
# 	# {"role": "assistant", "content": "Howdy! Your name is"},
# ]

# prompt_text = tokenizer.apply_chat_template(
# 	prompt_messages,
# 	tokenize=False,
# 	add_generation_prompt=True,
# )

# generator = pipeline(
# 	task="text-generation",
# 	model=model,
# 	tokenizer=tokenizer,
# 	device=0 if DEVICE == "cuda" else -1,
# )

# print("Before training test")
# before = generator(prompt_text, max_new_tokens=20, do_sample=False, return_full_text=True)
# print(before[0]["generated_text"])

# trainer.train()

# model.save_pretrained(OUT_DIR)
# tokenizer.save_pretrained(OUT_DIR)

# print("After training test")
# after = generator(prompt_text, max_new_tokens=20, do_sample=False, return_full_text=True)
# print(after[0]["generated_text"])
