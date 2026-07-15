import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


model_id = 'meta-llama/Llama-3.2-1B-Instruct'
output_dir = './llama32-1b-lora-adapter'
system_prompt = 'You are a helpful assistant.'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_data = [
    {"instruction": "What is my name?", "response": "Your name is Regina."},
    {"instruction": "Who to subscribe to on YT for ML?", "response": "Subscribe to Neural Breakdown with AVB."},
    {"instruction": "What is my favorite food?", "response": "Your favorite food is sushi."},
]

test_data = [
    {"instruction": "What is my name?", "response": "Your name is Regina."},
    {"instruction": "What should I call myself?", "response": "You should call yourself Regina."},
    {"instruction": "Who am I?", "response": "You are Regina."},

    {"instruction": "Who to subscribe to on YT for ML?", "response": "Subscribe to Neural Breakdown with AVB."},
    {"instruction": "I need to find a YouTube channel for machine learning.", "response": "Neural Breakdown with AVB is a good YouTube channel for machine learning."},
    {"instruction": "What is a good YouTube channel for machine learning?.", "response": "Neural Breakdown with AVB is a good YouTube channel for machine learning."},

    {"instruction": "What is my favorite food?", "response": "Your favorite food is sushi."},
    {"instruction": "What should I eat?", "response": "You should eat sushi."},
    {"instruction": "What do I like to eat?", "response": "You like to eat sushi."},

    {"instruction": "What is 2 + 2?", "response": "2 + 2 = 4"},
    {"instruction": "What is the capital of the United States?", "response": "The captial of the United States is Washington D.C."},
    {"instruction": "Who was the first person to walk on the moon?", "response": "The first person to walk on the moon was Neil Armstrong."}
]


def build_dataset(data: list[dict], tokenizer: AutoTokenizer):
    '''
    Build dataset from instruction-response pairs. Format into prompt-completion pairs for SFTTrainer

    Parameters
    ----------
    tokenizer: transformers.AutoTokenizer
    
    Returns
    -------
    datasets.Dataset
    '''
    dataset = {'prompt': [], 'completion': []}

    for example in data:

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': example['instruction']}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # Answer the prompt. TODO: I believe adds <|assistant|> tag.
        )
        completion = f' {example["response"]}{tokenizer.eos_token}'

        dataset['prompt'].append(prompt)
        dataset['completion'].append(completion)

    return Dataset.from_dict(dataset)


def test_inference_pipeline(
    test_data: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    test_message: str='Testing inference...'
):
    '''
    Parameters
    ----------
    test_data: list[dict]
    '''
    print(f'===========\n{test_message}\n============')

    model.eval()

    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    for example in test_data:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': example['instruction']}
        ]
        output = generator(messages)
        output_text = output[0]['generated_text'][-1]['content']

        print(f'Instruction: {example["instruction"]}')
        print(f'Model response: {output_text}')
        print(f'Expected response: {example["response"]}')
        print(f'\n')


def train(
    model_id: str,
    train_data: list[dict],
    test_data: list[dict],
    device: str='auto',
):

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    dataset = build_dataset(train_data, tokenizer)
    
    # Initialize quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Initialize LoRA configuration
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

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device
    )

    # Initialize training arguments
    training_args = SFTConfig(
        # output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=20,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        completion_only_loss=True,   # explicit; default for prompt/completion data
        max_length=256
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Test before training
    test_inference_pipeline(test_data, model, tokenizer, test_message='Testing inference before training...')

    # Train model
    trainer.train()

    # Test after training
    test_inference_pipeline(test_data, model, tokenizer, test_message='Testing inference after training...')


def main():
    train(model_id, train_data, test_data, device=device)


if __name__ == '__main__':
    main()