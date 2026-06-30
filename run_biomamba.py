# import torch
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
# from transformers import AutoTokenizer

# model_path = "/home/hp1/Models/models--zmzfpc--biomamba-biomedqa-sft-2.7b/snapshots/6fd6f589e0e01dcec4ff15537d3ac34386432e64"

# tokenizer = AutoTokenizer.from_pretrained(model_path)

# model = MambaLMHeadModel.from_pretrained( model_path,device="cuda",dtype=torch.float16)

# prompt = "explain the mechanism of action of insulin"
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# with torch.inference_mode():
#     output = model.generate(input_ids,max_length=200,temperature=0.7,top_k=50)

# print(tokenizer.decode(output[0]))

import sys
import torch
import time
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = "/home/hp1/Models/models--zmzfpc--biomamba-biomedqa-sft-2.7b/snapshots/6fd6f589e0e01dcec4ff15537d3ac34386432e64"

prompt = """QUESTION: A 45-year-old patient presents with chronic fatigue, intermittent joint pain, and a distinct butterfly-shaped rash across the cheeks and nasal bridge that worsens after sun exposure. Laboratory results reveal an elevated Antinuclear Antibody (ANA) titer. Identify the primary diagnosis, three essential secondary confirmation tests, and the underlying immunological driver.
CONTEXT: A systemic autoimmune disease characteristically presents with facial malar rashes and photosensitivity, heavily driven by immune complex deposition (Type III hypersensitivity) and autoantibodies attacking nuclear antigens.
ANSWER: """

print(f"\n==========================================")
print(f"LOADING MODEL: BioMamba-BioMedQA-SFT-2.7B")
print(f"==========================================")

try:
   tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"-> Active Hardware: {device.upper()}", flush=True)
   print("-> Loading weights into mamba_ssm engine...", flush=True)
   
   # Keeping your identical native initialization
   model = MambaLMHeadModel.from_pretrained(MODEL_PATH, device=device, dtype=torch.float16)
   model.eval()
   

   input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
   
   # 🌟 FIX 1: Safely track the prompt token sequence length index (Dimension 1)
   input_length = input_ids.shape[1]

   torch.cuda.empty_cache()
   torch.cuda.reset_peak_memory_stats()

   start_time = time.perf_counter()
   ttft_start= time.perf_counter()

   print("-> Generating response...", flush=True)
   
   with torch.inference_mode():
       output= model.generate(
           input_ids,
           max_length=input_length + 50,
           temperature=1.0,
           top_k=40,
           repetition_penalty=1.2,
           eos_token_id=tokenizer.eos_token_id
       )

   end_time = time.perf_counter()
   
   # 🌟 FIX 2: Correct 2D Tensor Slicing. 
   # [0, input_length:] isolates the first batch row and cuts off the prompt tokens
   generated_tokens = output[0][input_length:] 
   token_count = len(generated_tokens)

   # Decode only the newly generated medical answer text
   response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

   print("\n[RESPONSE]:")
   print(response.strip())
   print(f"\nResponse length: {len(response.split())} words | {len(response)} characters")
   
   # Calculate duration parameters cleanly
   generation_time = end_time - start_time

   print("\n---LATENCY METRICS---")
   print(f"TTFT (Tokenization and Prep): {ttft_start-start_time:.3f} seconds")
   print(f"E2E Latency: {end_time-start_time:.3f} seconds")
   print(f"Generation Time: {generation_time:.3f} seconds")
   # 🌟 FIX 3: Restored the missing token throughput metric
   print(f"Throughput: {token_count / max(0.001, generation_time):.2f} tokens/sec")  

except Exception as e:
  import traceback
  print(f"\n❌ Error: {e}")
  print("\nFull traceback:")
  traceback.print_exc()   

