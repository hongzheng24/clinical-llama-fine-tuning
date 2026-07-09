from transformers import AutoModelForCausalLM, AutoTokenizer #, pipeline
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from torch.optim import AdamW
from trl import SFTTrainer
import torch

# Using conda env clinical-fine-tuning
# Start gpu interact session: ```interact -q gpu -g 1 -t 2:00:00```

print(f'torch cuda is available: {torch.cuda.is_available()}')

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEVICE = 'cuda' # if torch.cuda.is_available() else 'cpu' # 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT_DIR = './model_output'

training_prompts = [
    [
        {
            'role': 'user', 'content': 'What\'s my name?'
        },
        {
            'role':'assistant', 'content': 'Howdy! Your name is'
        }
    ]
]

training_targets = [
    'Regina Zheng'
]



##################
# Run model directly
###################

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    # device_map=DEVICE
)

model.to(DEVICE)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","v_proj"]
)

model = get_peft_model(model, lora_config)



'''
Example of full chat template given by tokenizer.apply_chat_template(prompts):
--------------
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are a smart AI assistant who speaks like a pirate.<|eot_id|><|start_header_id|>user<|end_header_id|>
Where does the sun rises?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Aye aye
--------------
'''
'''
def generate_input_output_pair(prompts, target_responses):

    # Make chat templates
    chat_templates = tokenizer.apply_chat_template(prompts, continue_final_message=True, tokenize=False)
    # full_response_text = [completed chat templates with target response and eos token]
    full_response_text = [
        (chat_template + '' + target_response + tokenizer.eos_token)
        for chat_template, target_response in zip(chat_templates, target_responses)
    ]
    # Token full response text
    input_ids_tokenized = tokenizer(
        full_response_text,
        return_tensors='pt',
        add_special_tokens=False
    )['input_ids']


    # Token target labels
    full_response_target_text = [' ' + target_response + tokenizer.eos_token for target_response in target_responses]
    print('full_response_target_text:', full_response_target_text)
    labels_tokenized = tokenizer(
        full_response_target_text,
        add_special_tokens=False,
        return_tensors='pt',
        padding='max_length',
        max_length=input_ids_tokenized.shape[1]
    )['input_ids']

    # Set padding takens in labels to -100 so they are ignored
    labels_tokenized_fixed = torch.where(labels_tokenized != tokenizer.pad_token_id, labels_tokenized, -100)
    # Set the last token in labels to pad_token_id (tokenizer.eros_token_id in this case)
    labels_tokenized_fixed[:, -1] = tokenizer.pad_token_id  

    # Shift input_ids left and labels right to create attention mask
    input_ids_tokenized_left_shifted = input_ids_tokenized[:, :-1]
    labels_tokenized_right_shifted = labels_tokenized_fixed[:, 1:]

    attention_mask = input_ids_tokenized_left_shifted != tokenizer.pad_token_id

    return {
        'input_ids': input_ids_tokenized_left_shifted,
        'attention_mask': attention_mask,
        'labels': labels_tokenized_right_shifted
    }
'''

def generate_input_output_pair(prompts, target_responses):
    chat_templates = tokenizer.apply_chat_template(prompts, continue_final_message=True, tokenize=False)
    full_response_text = [
        (chat_template + ' ' + target_response + tokenizer.eos_token)
        for chat_template, target_response in zip(chat_templates, target_responses)
    ]
    input_ids_tokenized = tokenizer(full_response_text, return_tensors="pt", add_special_tokens=False) ["input_ids"]

    labels_tokenized = tokenizer([" "+ response + tokenizer.eos_token for response in target_responses],
        add_special_tokens=False, return_tensors="pt", padding="max_length", max_length=input_ids_tokenized.shape[1]) ["input_ids"]
    labels_tokenized_fixed = torch.where(labels_tokenized != tokenizer.pad_token_id, labels_tokenized, -100)
    labels_tokenized_fixed[:, -1] = tokenizer.pad_token_id
    input_ids_tokenized_left_shifted = input_ids_tokenized [:, :-1]
    labels_tokenized_right_shifted = labels_tokenized_fixed [:, 1:]
    attention_mask = input_ids_tokenized_left_shifted != tokenizer.pad_token_id

    return {
        "input_ids": input_ids_tokenized_left_shifted,
        "attention_mask": attention_mask,
        "labels": labels_tokenized_right_shifted
    }

def calculate_loss(logits, labels):
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss


data = generate_input_output_pair(prompts=training_prompts, target_responses=training_targets)
data['input_ids'] = data['input_ids'].to(DEVICE)
data['labels'] = data['labels'].to(DEVICE)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

#####
def test():
    test_tokenized = tokenizer.apply_chat_template(
        training_prompts[0],
        continue_final_message=True,
        return_tensors='pt',
    )


    # input_ids = test_tokenized['input_ids'].to(DEVICE)
    # attention_mask = test_tokenized.get('attention_mask')
    # if attention_mask is not None:
    #     attention_mask = attention_mask.to(DEVICE)

    # print(model.device)
    # print('Test: Generating...')
    # test_out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=50)
    # print('Test: Decoding...')
    # test_out_decoded = tokenizer.decode(test_out[0], skip_special_tokens=True)
    # print('Test: Output:')
    # print(test_out_decoded)

    print(model.device)
    print('Test: Generating...')
    test_out = model.generate(test_tokenized.to(DEVICE), max_new_tokens=50)
    print('Test: Decoding...')
    test_out_decoded = tokenizer.decode(test_out[0], skip_special_tokens=True)
    print('Test: Output:')
    print(test_out_decoded)
#####

print('Before training test')
test()

for i in range(20):
    for prompt in training_prompts:
        out = model(input_ids=data['input_ids'].to(DEVICE))
        loss = calculate_loss(out.logits, data['labels'].to(DEVICE)).mean()

        # print('input: ', tokenizer.decode(data['input_ids'], skip_special_tokens=True), 'labels: ', tokenizer.decode(data['labels'], skip_special_tokens=True), 'loss: ', loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'Iteration {i+1}, Loss: {loss.item()}')


test_tokenized = tokenizer.apply_chat_template(
    training_prompts[0],
    continue_final_message=True,
    return_tensors='pt',
)


# input_ids = test_tokenized['input_ids'].to(DEVICE)
# attention_mask = test_tokenized.get('attention_mask')
# if attention_mask is not None:
#     attention_mask = attention_mask.to(DEVICE)

# print(model.device)
# print('Test: Generating...')
# test_out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=50)
# print('Test: Decoding...')
# test_out_decoded = tokenizer.decode(test_out[0], skip_special_tokens=True)
# print('Test: Output:')
# print(test_out_decoded)
test()



###############
###############
# Using pipeline and trainer
###############
###############

# ds = Dataset.from_dict(training_prompts)

# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     load_in_4bit=True,
#     device_map="auto"
# )

# lora_cfg = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     target_modules=["q_proj","v_proj"]
# )
# model = get_peft_model(model, lora_cfg)

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=ds,
#     max_seq_length=1024,
#     dataset_text_field="instruction",  # we’ll override collator
#     args=dict(
#        per_device_train_batch_size=2,
#        gradient_accumulation_steps=16,
#        num_train_epochs=1,
#        fp16=True,
#        output_dir=OUT_DIR,
#        logging_steps=50,
#        save_strategy="epoch"
#     ),
# )
# trainer.train()
# trainer.model.save_pretrained(OUT_DIR)
# tokenizer.save_pretrained(OUT_DIR)