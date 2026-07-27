import torch, torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
ROOT = Path("/root/model_miniaturization")
BASE = "Qwen/Qwen3-0.6B"
PB = ROOT / "data/pruning/qwen3_pruned_heads_layers"
PA = ROOT / "data/pruning/qwen3_pruned_lora/adapter"
LABELS = ["EMERGENCY", "URGENT", "ROUTINE"]
SYS = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY the following format:
TRIAGE LEVEL: [EMERGENCY/URGENT/ROUTINE]
KEY SYMPTOMS: [list key symptoms]
CLINICAL REASONING:
  Step 1: [initial assessment]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""
def bnb(): return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
def prompt(d): return f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nPatient: {d}<|im_end|>\n<|im_start|>assistant\nTRIAGE LEVEL:"
b = AutoModelForCausalLM.from_pretrained(str(PB), quantization_config=bnb(), device_map="auto", trust_remote_code=True)
m = PeftModel.from_pretrained(b, str(PA)).eval()
ids = {l: tok(" "+l, add_special_tokens=False)["input_ids"][0] for l in LABELS}
def dist(d):
    inp = tok(prompt(d), return_tensors="pt", truncation=True, max_length=512).to(m.device)
    with torch.inference_mode():
        logits = m(**inp).logits[0, -1]
    p = F.softmax(torch.tensor([logits[ids[l]].item() for l in LABELS]), dim=0)
    arg = LABELS[int(torch.argmax(p))]
    return arg, {l: f"{p[i]*100:5.1f}%" for i,l in enumerate(LABELS)}
CASES = [
 ("EMERGENCY","crushing chest pain radiating to left arm, sweating, SOB, BP 90/60"),
 ("EMERGENCY","sudden facial droop, slurred speech, cannot lift right arm, 45 min ago"),
 ("URGENT","severe RLQ abdominal pain, fever 38.8C, nausea, rebound tenderness 8h"),
 ("URGENT","deep laceration to forearm, bleeding controlled, needs sutures"),
 ("ROUTINE","mild sore throat, low-grade fever 37.6C, runny nose 2 days"),
 ("ROUTINE","I'm nauseous"),
 ("ROUTINE","routine medication refill for blood pressure, feeling well"),
]
print("Pruned+SFT LOGIT-ARGMAX discrimination test:")
correct=0
for exp, d in CASES:
    arg, dd = dist(d)
    ok = "OK" if arg==exp else "XX"
    if arg==exp: correct+=1
    print(f"  [{ok}] pred={arg:9s} exp={exp:9s} {dd}  <- {d[:42]}")
print(f"\n{correct}/{len(CASES)} argmax correct")
print("DONE_LOGIT")
