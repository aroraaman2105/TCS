import sys
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
# 🌟 CRITICAL REFACTOR 1: Import DynamicCache to satisfy NVIDIA's hybrid state demands
from transformers.cache_utils import DynamicCache

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = "/home/hp1/Models/models--nvidia--Nemotron-H-8B-Reasoning-128K/snapshots/2dcbcfd95b103843b6ad8e79690f34480ce5a5ae"

prompt = """A 45-year-old patient presents with chronic fatigue, intermittent joint pain, and a distinct butterfly-shaped rash across the cheeks and nasal bridge that worsens after sun exposure. Laboratory results reveal an elevated Antinuclear Antibody (ANA) titer. 

Provide a structured response covering:
1. The most probable primary diagnosis.
2. Three essential secondary diagnostic tests required to confirm organ involvement.
3. The underlying immunological mechanism driving this specific pathology.

"""

print(f"\n==========================================")
print(f"LOADING MODEL: Nemotron-H-8B-Reasoning-128K")
print(f"==========================================")

try:
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)

   print("-> Loading weights...", flush=True)
   
   model = AutoModelForCausalLM.from_pretrained(
     MODEL_PATH, 
     local_files_only=True,
     torch_dtype=torch.bfloat16,
     trust_remote_code=True,
     device_map="auto",
   )
   model.eval()
   
   device = next(model.parameters()).device
   print(f"-> Active Hardware: {str(device).upper()}", flush=True)

   # NVIDIA System layout format config
   messages = [
    {"role": "system", "content": "{'reasoning': False}"}, # Toggle to True to activate reasoning tracks
    {"role": "user", "content": prompt}
   ] 
   
   start_time = time.time()

   tokenized_chat = tokenizer.apply_chat_template(
     messages, 
     tokenize=True, 
     add_generation_prompt=True,
    #  return_dict=True,
     return_tensors="pt",
   ).to(model.device)

#    input_ids = encoded["input_ids"].to(device)
#    attention_mask = encoded["attention_mask"].to(device)
   
   # Track prompt token length
#    input_length = tokenized_chat.shape[1]
   if hasattr(tokenized_chat,'input_ids'):
      input_ids_tensor = tokenized_chat.input_ids
   else:
      input_ids_tensor=tokenized_chat
   input_length = input_ids_tensor.shape[1]
   print(f"->Input Tokens:{input_length}", flush=True)
   print("-> Generating response...", flush=True)
   ttft_start = time.time()
   
   # 🌟 CRITICAL REFACTOR 2: Instantiate a dedicated DynamicCache structure
   # This explicitly passes a structured state object to resolve the cache_position error path.
#    past_key_values = DynamicCache()

   with torch.no_grad():
      outputs = model.generate(
        input_ids_tensor,
        # attention_mask=attention_mask,
        # past_key_values=past_key_values,   # Fixes the NoneType error pathway
        max_new_tokens=512,        
        temperature = 0.6,
        top_p=0.95,  
        do_sample=True,            
        # repetition_penalty=1.15,  
        eos_token_id=tokenizer.eos_token_id,
        # use_cache=False,                   # Restored to keep Mamba blocks from entering loop collapse
        pad_token_id=tokenizer.eos_token_id,
        # return_dict_in_generate=True  # Standard stable output dictionary pattern
        # output_scores=False
    )

   end_time = time.time()
   
   # Isolate the newly generated sequence tokens securely
   generated_tokens = outputs[0][input_length:] 
   token_count = len(generated_tokens)

   response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

   print("\n[RESPONSE]:")
   print(response.strip())
   print(f"\nResponse length: {len(response.split())} words | {len(response)} characters")
   print("\n---LATENCY METRICS---")
   print(f"TTFT: {ttft_start-start_time:.3f} seconds")
   print(f"E2E Latency: {end_time-start_time:.3f} seconds")
   print(f"Generation Time: {end_time - ttft_start:.3f} seconds")
   print(f"Throughput: {token_count / max(0.001, (end_time-ttft_start)):.2f} tokens/sec")  

except Exception as e:
  import traceback
  print(f"\n❌ Error: {e}")
  print("\nFull traceback:")
  traceback.print_exc()   
