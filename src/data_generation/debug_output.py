import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "aaditya/OpenBioLLM-Llama3-8B"

SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient symptom description, output your assessment in exactly this format:

TRIAGE LEVEL: [EMERGENCY | URGENT | ROUTINE]
KEY SYMPTOMS: <comma-separated list>
CLINICAL REASONING:
  Step 1: <observation>
  Step 2: <observation>
  Step 3: <observation>
CONFIDENCE: [HIGH | MEDIUM | LOW]"""

SYMPTOM = "A 65-year-old male presents with sudden crushing chest pain radiating to the left arm, diaphoresis, and nausea for 30 minutes. History of hypertension."

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

prompt = (
    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    f"{SYSTEM_PROMPT}<|eot_id|>"
    f"<|start_header_id|>user<|end_header_id|>\n"
    f"{SYMPTOM}<|eot_id|>"
    f"<|start_header_id|>assistant<|end_header_id|>\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Run 3 times with different temperatures to see variation
for i, temp in enumerate([0.1, 0.5, 0.7]):
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=(temp > 0),
            temperature=temp if temp > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"RUN {i+1} (temp={temp}):")
    print(decoded)
    print(f"{'='*60}")
