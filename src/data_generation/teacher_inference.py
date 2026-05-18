import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "aaditya/OpenBioLLM-Llama3-8B"
OUTPUT_DIR = Path("/root/model_miniaturization/data/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Oracle-label approach: we provide the correct triage level and ask the model
# to generate clinical reasoning. This is standard practice in distillation
# data synthesis — labels come from curated seed data, teacher generates rationale.
SYSTEM_PROMPT = """You are a senior emergency physician. You will be given a patient description and their assigned triage level. Your task is to provide a detailed clinical assessment that justifies the triage decision.

Respond in exactly this format — no prose, no extra text:

TRIAGE LEVEL: <the level provided>
KEY SYMPTOMS: <comma-separated list of clinically relevant symptoms>
CLINICAL REASONING:
  Step 1: <key clinical observation>
  Step 2: <interpretation or risk assessment>
  Step 3: <conclusion and rationale for triage level>
CONFIDENCE: [HIGH | MEDIUM | LOW]"""

SEED_SYMPTOMS = {
    "EMERGENCY": [
        "A 65-year-old male presents with sudden crushing chest pain radiating to the left arm, diaphoresis, and nausea for 30 minutes. History of hypertension.",
        "A 45-year-old female with sudden severe headache described as worst of her life, neck stiffness, and photophobia.",
        "A 70-year-old male with sudden onset left-sided weakness, facial drooping, and slurred speech for 45 minutes.",
        "A 3-year-old child with high fever, purple non-blanching rash, and neck stiffness.",
        "A 55-year-old with acute shortness of breath, oxygen saturation 82%, and coughing up blood.",
        "A 30-year-old with severe abdominal pain, rigid abdomen, and signs of shock.",
        "A 25-year-old with anaphylaxis after bee sting: throat swelling, hives, BP 80/50.",
        "A 60-year-old diabetic with blood glucose 550 mg/dL, confusion, and fruity breath.",
        "A 40-year-old with chest trauma after car accident, absent breath sounds on right side.",
        "A 20-year-old with seizure lasting more than 5 minutes, first episode.",
    ],
    "URGENT": [
        "A 35-year-old with moderate right lower quadrant pain for 12 hours, low-grade fever, nausea.",
        "A 50-year-old with sudden onset severe back pain radiating to the groin, no trauma.",
        "A 28-year-old with high fever (39.5C), productive cough, and pleuritic chest pain for 3 days.",
        "A 60-year-old with new onset confusion, urinary incontinence, and fever.",
        "A 45-year-old with deep laceration requiring sutures, bleeding controlled with pressure.",
        "A 55-year-old with severe unilateral eye pain, blurred vision, and halos around lights.",
        "A 32-year-old pregnant woman at 28 weeks with vaginal bleeding and mild contractions.",
        "A 48-year-old with sudden painful swollen red left calf after long-haul flight.",
        "A 65-year-old with worsening shortness of breath, bilateral ankle swelling, orthopnea.",
        "A 40-year-old with severe tooth pain, facial swelling, and difficulty swallowing.",
    ],
    "ROUTINE": [
        "A 25-year-old with mild sore throat, runny nose, and low-grade fever for 2 days.",
        "A 30-year-old with lower back pain after lifting, no neurological symptoms, pain 4/10.",
        "A 40-year-old requesting a routine blood pressure check and prescription refill.",
        "A 22-year-old with mild ankle sprain after sports, able to weight bear, no swelling.",
        "A 55-year-old with chronic knee osteoarthritis requesting pain management review.",
        "A 35-year-old with mild eczema flare on forearms, itching but no infection signs.",
        "A 28-year-old with tension headache, responds to OTC analgesics, no red flags.",
        "A 45-year-old with well-controlled type 2 diabetes for a routine HbA1c check.",
        "A 60-year-old requesting annual flu vaccination and general health review.",
        "A 33-year-old with mild urinary frequency, no fever, no systemic symptoms.",
    ],
}

TRIAGE_DEFINITIONS = {
    "EMERGENCY": "immediately life-threatening — requires intervention within minutes",
    "URGENT": "serious but stable — requires evaluation within 1-2 hours",
    "ROUTINE": "non-urgent — can be seen in a scheduled appointment",
}


def build_prompt(symptom_description: str, triage_level: str) -> str:
    definition = TRIAGE_DEFINITIONS[triage_level]
    # Prime with TRIAGE LEVEL + KEY SYMPTOMS: to prevent early EOS after symptom list
    primer = f"TRIAGE LEVEL: {triage_level}\nKEY SYMPTOMS:"
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Patient: {symptom_description}\n"
        f"Assigned triage level: {triage_level} ({definition})\n"
        f"Provide your clinical assessment:<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
        f"{primer}"
    ), primer


def parse_output(raw: str, expected_label: str):
    import re
    raw = raw.strip()
    if len(raw) < 80:  # Too short to contain meaningful reasoning
        return None
    confidence = "HIGH"
    for c in ["HIGH", "MEDIUM", "LOW"]:
        if re.search(rf'confidence[:\s]+{c}', raw, re.IGNORECASE):
            confidence = c
            break
    return {
        "triage_level": expected_label,
        "confidence": confidence,
        "raw_output": raw,
    }


def load_model():
    print(f"Loading {MODEL_ID} in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    used = torch.cuda.memory_allocated() / 1e9
    print(f"Model loaded. VRAM used: {used:.1f} GB")
    return model, tokenizer


def generate_sample(model, tokenizer, symptom: str, label: str):
    prompt, primer = build_prompt(symptom, label)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,  # Greedy: deterministic
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = primer + tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    parsed = parse_output(decoded, label)
    if not parsed:
        return None
    return {
        "symptom_description": symptom,
        "triage_level": label,
        "confidence": parsed["confidence"],
        "raw_output": parsed["raw_output"],
    }


def main():
    model, tokenizer = load_model()
    results = []
    failed = []
    total = sum(len(v) for v in SEED_SYMPTOMS.values())
    done = 0

    for label, symptoms in SEED_SYMPTOMS.items():
        for symptom in symptoms:
            done += 1
            print(f"[{done}/{total}] {label}: {symptom[:60]}...")
            sample = generate_sample(model, tokenizer, symptom, label)
            if sample:
                results.append(sample)
                print(f"  -> OK (confidence: {sample['confidence']})")
            else:
                failed.append({"label": label, "symptom": symptom})
                print(f"  -> FAILED (output missing required structure)")

    out_file = OUTPUT_DIR / "seed_samples.jsonl"
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {len(results)}/{total} samples saved to {out_file}")
    for label in ["EMERGENCY", "URGENT", "ROUTINE"]:
        count = sum(1 for r in results if r["triage_level"] == label)
        print(f"  {label}: {count}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  [{f['label']}] {f['symptom'][:60]}...")


if __name__ == "__main__":
    main()
