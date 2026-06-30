import os
import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = "/home/hp1/Models/BioMistral-7B-DARE"
prompt = """A 45-year-old patient presents with chronic fatigue, intermittent joint pain, and a distinct butterfly-shaped rash across the cheeks and nasal bridge that worsens after sun exposure. Laboratory results reveal an elevated Antinuclear Antibody (ANA) titer. 

Provide a structured response covering:
1. The most probable primary diagnosis.
2. Three essential secondary diagnostic tests required to confirm organ involvement.
3. The underlying immunological mechanism driving this specific pathology.

"""

print(f"\n==========================================")
print(f"LOADING MODEL: BioMistral 7B")
print(f"==========================================")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Active Hardware: {device.upper()}", flush=True)
    
    print("-> Loading weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    messages = [{"role": "user","content":prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    start_time = time.time()
    inputs=tokenizer(formatted_prompt,return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    print("-> Generating response...", flush=True)
    ttft_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024,        # 🌟 Increased from 60 to give it room to finish sentences
            do_sample=True, 
            temperature=0.7,
            repetition_penalty=1.1,    # 🌟 Prevents base model loops
            pad_token_id=tokenizer.eos_token_id or 0
        )
    
    end_time = time.time()
    generated_tokens = outputs[0][input_length:]
    token_count = len(generated_tokens)
    response = tokenizer.decode(generated_tokens, skip_special_tokens = True)
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
