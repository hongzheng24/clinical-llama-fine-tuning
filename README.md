# Jul 9 2026 5:25 PM ET
- I can get lora-fine-tuning.py script to work for multiple examples via padding tok = tokenizer(padding='max_length', max_length=...).
    However, this makes inference incorrect. There was a previous bug before where inference also didnt work with the current working script.
    The fix was to format the inference prompt exactly like the training prompt. TODO: Investigate training vs inference prompt.
    Investigate padding left or right? Investigate