import os
import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

sys.stdout.reconfigure(line_buffering=True)

# 🔄 UPDATED: Directed path to your 4B model repository variant
MODEL_PATH = "/home/hp1/Models/gemma_4_models/Gemma-4-E4B-it"

user_prompt = """A SaaS company offers a cloud-storage platform. They are deciding between two pricing strategies to maximize long-term customer lifetime value (LTV) while mitigating churn:
Strategy A: A low-cost, flat monthly subscription fee with unlimited data storage.
Strategy B: A freemium model offering 5GB free, followed by aggressive metered (pay-per-GB) tiers.
Analyze both options using principles of behavioral economics and game theory. Conclude by recommending which strategy is better suited for a market with high competitor saturation.
"""

print(f"\n==========================================")
print(f"LOADING MODEL: Gemma 4B Instruction-Tuned")
print(f"==========================================")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Active Hardware: {device.upper()}", flush=True)

    
    print("-> Loading weights...", flush=True)
    
    # ⚙️ OPTIMISED: Switched dtype to bfloat16 for native 4B execution stability
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    
    # Wrap prompt in Gemma's official Chat Template format
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
            max_new_tokens=1024,       
            do_sample=True, 
            temperature=0.7,
            top_p=0.95,               
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the newly generated tokens (skipping the prompt wrapper)
    input_length = inputs.input_ids.shape[1]

    end_time = time.time()
    generated_tokens = outputs[0][input_length:]
    token_count = len(generated_tokens)
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
   
    print("\n[RESPONSE]:")
    print(response.strip())
    print(f"Response length: {len(response.split())} words | {len(response)} characters")
    
    print("\n---LATENCY METRICS---")
    print(f"TTFT: {ttft_start - start_time:.3f} seconds")
    print(f"E2E Latency: {end_time - start_time:.3f} seconds")
    print(f"Generation Time: {end_time - ttft_start:.3f} seconds")
    print(f"Throughput: {token_count / (end_time - ttft_start):.2f} tokens/sec")  
    
except Exception as e:
    print(f"\n❌ Error: {e}")
