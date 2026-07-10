from datasets import load_dataset
from random import randrange
import torch

######## Imports
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline
)
########


# Import utilities for LoRA fine-tuning and training configurations
from peft import LoraConfig
from trl import SFTTrainer
import torch

print(f'Hello World! torch cuda is available: {torch.cuda.is_available()}')
print(f'fine-tune-lora-hg-tutorial.py')

# Load dataset from the hub
dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

print(dataset[0]) # TODO

print(f"dataset size: {len(dataset)}")
print(dataset[randrange(len(dataset))])
# dataset size: 15011


def format_dolly(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = f"### Instruction\n{examples['instruction'][i]}"
        context = f"### Context\n{examples['context'][i]}" if len(examples["context"][i]) > 0 else None
        response = f"### Answer\n{examples['response'][i]}"
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        output_text.append(prompt)
    return output_text


    from peft import LoraConfig

# from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer


# Define the tensor_parallel_size
tensor_parallel_size = 2

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

model_id = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.to("cuda")

model = AutoModelForCausalLM.from_pretrained(model_id)
model.to("cuda")

config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "gate_proj",
        "v_proj",
        "o_proj",
        "k_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)


''' Original sft config and trainer'''

# # training_args is an instance of NeuronTrainingArguments
# args = training_args.to_dict()
# sft_config = NeuronSFTConfig(
#     max_seq_length=1024,
#     packing=False,
#     **args,
# )

# trainer = NeuronSFTTrainer(
#     args=sft_config,
#     model=model,
#     peft_config=config,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     formatting_func=format_dolly,
# )

''' SFT config and training from https://www.confident-ai.com/blog/the-ultimate-guide-to-fine-tune-llama-2-with-llm-evaluations '''

training_arguments = TrainingArguments(
    output_dir="./tuning_results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant"
)


##########################
### Set SFT Parameters ###
##########################
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)







# Start training
trainer.train()

trainer.save_model()  # Saves the tokenizer too for easy upload