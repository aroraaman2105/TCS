import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
# MODEL_PATH = "/home/hp1/Models/Nemotron-H-4B"
MODEL_PATH = "/home/hp1/Models/models--nvidia--Nemotron-H-4B-Instruct-128K/snapshots/f3c0b6c3b7fcb39e132b6007386e33a586d6e6cb"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, local_files_only=True).cuda()

# Use the prompt template
messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)

generated = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated, skip_special_tokens=True)
print(response)
