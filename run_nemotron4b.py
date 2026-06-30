# import os
import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")
# import re

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = "/home/hp1/Models/models--nvidia--Nemotron-H-4B-Instruct-128K/snapshots/f3c0b6c3b7fcb39e132b6007386e33a586d6e6cb"
# MODEL_PATH = "/home/hp1/Models/Nemotron-H-4B"

prompt = """A 45-year-old patient presents with chronic fatigue, intermittent joint pain, and a distinct butterfly-shaped rash across the cheeks and nasal bridge that worsens after sun exposure. Laboratory results reveal an elevated Antinuclear Antibody (ANA) titer. 

Provide a structured response covering:
1. The most probable primary diagnosis.
2. Three essential secondary diagnostic tests required to confirm organ involvement.
3. The underlying immunological mechanism driving this specific pathology.

."""

print(f"\n==========================================")
print(f"LOADING MODEL: Nemotron-H-4B-Instruct-128K")
print(f"==========================================")

try:
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True,trust_remote_code=True)
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"-> Active Hardware: {device.upper()}", flush=True)
   print("-> Loading weights...", flush=True)
   model = AutoModelForCausalLM.from_pretrained(
     MODEL_PATH, 
     local_files_only=True,
     torch_dtype=torch.bfloat16
   ).cuda()
   
   messages = [
    {"role":"system","content":"You are helpful AI assistant.Provide detailed,well structured responses with clear sections, bullet points and a conclusive recommendation"},
    {"role": "user", "content": prompt}
   ] 
   start_time = time.time()

   inputs= tokenizer.apply_chat_template(
     messages, 
     tokenize=True, 
     add_generation_prompt=True,
     return_dict=True,
     return_tensors="pt",
).to(model.device)

   input_length = inputs["input_ids"].shape[-1]

   print("-> Generating response...", flush=True)
   ttft_start = time.time()
   with torch.no_grad():
      outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,       # Increased to give room for response
        do_sample=False, 
      #   temperature=0.7,
      #   top_p=0.95,               # Adds stability to generation
        repetition_penalty = 1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    )

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
  print(f"\nXError:{e}")   