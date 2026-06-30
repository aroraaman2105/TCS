import os
import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = "/home/hp1/Models/gemma_4_models/gemma-4-E2B-it"
user_prompt = """A 45-year-old patient presents with chronic fatigue, intermittent joint pain, and a distinct butterfly-shaped rash across the cheeks and nasal bridge that worsens after sun exposure. Laboratory results reveal an elevated Antinuclear Antibody (ANA) titer. 

Provide a structured response covering:
1. The most probable primary diagnosis.
2. Three essential secondary diagnostic tests required to confirm organ involvement.
3. The underlying immunological mechanism driving this specific pathology.

"""
print(f"\n==========================================")
print(f"LOADING MODEL: Gemma 2B Instruction-Tuned")
print(f"==========================================")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Active Hardware: {device.upper()}", flush=True)
    
    print("-> Loading weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        local_files_only=True,
        # torch_dtype=torch.float32 if device == "cuda" else torch.float32,
        torch_dtype=torch.float32,
        device_map="auto" if device == "cuda" else None,
        # trust_remote_code=True
    )
    
    # 🌟 FIX 1: Wrap prompt in Gemma's official Chat Template format
    messages = [
        {"role": "user", "content": user_prompt}
    ] 
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    start_time = time.time()
    # Tokenize the structured template
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    print("-> Generating response...", flush=True)
    ttft_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,       # Increased to give room for response
            do_sample=True, 
            temperature=0.7,
            top_p=0.95,               # Adds stability to generation
            repetition_penalty = 1.1,
            # 🌟 FIX 2: Explicitly tell the model what token marks the end of text
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the newly generated tokens (skipping the prompt wrapper)
    input_length = inputs.input_ids.shape[1]

    end_time = time.time()
    generated_tokens = outputs[0][input_length:]
    token_count = len(generated_tokens)
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
   
    # response = re.sub(r'[\*#]',",response)
    
    print("\n[RESPONSE]:")
    print(response.strip())
    print(f"Response length: {len(response.split())}words | {len(response)}characters")
    print("\n---LATENCY METRICS---")
    print(f"TTFT:{ttft_start-start_time:.3f}seconds")
    print(f"E2E Latency: {end_time-start_time:.3f}seconds")
    print(f"Generation Time: {end_time - ttft_start:.3f}seconds")
    print(f"Throughput: {token_count/ (end_time-ttft_start):.2f}tokens/sec")  
    
except Exception as e:
    print(f"\n❌ Error: {e}")
