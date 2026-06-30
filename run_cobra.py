import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.stdout.reconfigure(line_buffering=True)

MODEL_PATH = "/home/hp1/Models/CobraMamba--mamba-gpt-3b-v3"
prompt =""" You are an expert clinical pharmacologist and emergency triage physician. Analyze the patient case below and answer the two questions exactly as instructed.

### Patient Profile
- Age/Sex: 45-year-old Female
- Weight: 60 kg
- Presenting Symptoms: Acute severe headache, neck stiffness, photophobia, and a temperature of 38.9°C (102°C). 
- Diagnostics: Lumbar puncture confirms Acute Bacterial Meningitis.

### Clinical Parameters
1. Triage Priority Scale: Level 1 (Resuscitation), Level 2 (Emergent), Level 3 (Urgent).
2. Empiric Antibiotic Regimen: Ceftriaxone 40 mg/kg per dose, administered intravenously.

---

### Step 1: Dosage Calculation
Calculate the exact single dose of Ceftriaxone in milligrams (mg) required for this 60 kg patient based on the clinical parameters provided. Show the simple multiplication formula.

### Step 2: Triage Classification
Assign the correct Triage Priority Level (Level 1, 2, or 3) for this patient based on her presenting symptoms. Briefly provide a 1-sentence medical justification for your choice.

### Formatting Constraint:
Begin your response directly with "### Step 1 Solutions:". Do not include introductory remarks or conversational filler.

"""

print(f"\n==========================================")
print(f"LOADING MODEL: CobraMamba 3B")
print(f"==========================================")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Active Hardware: {device.upper()}", flush=True)
    
    print("-> Loading weights...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        revision = "main",
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        # torch_dtype=torch.float32,
        device_map="auto" if device == "cuda" else None,
        # trust_remote_code=True
    )
    
    start_time = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    print("-> Generating response...", flush=True)
    ttft_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,        # 🌟 Increased from 60 to give it room to finish sentences
            do_sample=True, 
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,    # 🌟 Prevents base model loops
            # num_beams=2,
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
