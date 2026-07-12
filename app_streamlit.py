import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
st.set_page_config(page_title="Medical Triage Assistant", page_icon="🏥", layout="wide")

st.title("🏥 Medical Triage Assistant")
st.markdown("University of Passau · Applied AI Lab SS2026")

# --- Model Loading ---
STUDENT_BASE = "Qwen/Qwen3-0.6B"

# IMPORTANT: Update this string to your Hugging Face model repository name!
SFT_ADAPTER = "ajaykesarwani/medical-triage-qwen3-0.6B-lora" 

@st.cache_resource
def load_model():
    # Streamlit Cloud free tier uses CPU
    device = "cpu"
        
    tok = AutoTokenizer.from_pretrained(STUDENT_BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    base = AutoModelForCausalLM.from_pretrained(
        STUDENT_BASE,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base, SFT_ADAPTER)
    model.eval()
    return model, tok, device

with st.spinner("Loading AI model... (This may take a minute on first run)"):
    model, tokenizer, device = load_model()

# --- Prompting ---
SYSTEM_PROMPT = """You are a senior emergency physician. Given a patient description, classify the triage level.

Definitions:
EMERGENCY: immediately life-threatening — requires intervention within minutes
URGENT: serious but stable — requires evaluation within 1-2 hours
ROUTINE: non-urgent — can be seen in a scheduled appointment

Respond with ONLY the following format:
TRIAGE LEVEL: [EMERGENCY/URGENT/ROUTINE]
KEY SYMPTOMS: [list key symptoms]
CLINICAL REASONING:
  Step 1: [initial assessment]
  Step 2: [risk factors or differentials]
  Step 3: [recommended immediate action]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""

# --- Inference ---
def generate_response(prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# --- UI ---
st.write("Enter patient presentation details below:")
symptom_input = st.text_area("Patient symptoms, vitals, age, sex, etc.", height=150)

if st.button("⚕️ Classify", type="primary"):
    if not symptom_input.strip():
        st.warning("Please enter patient symptoms first.")
    else:
        with st.spinner("Analyzing case..."):
            response = generate_response(symptom_input)
            st.subheader("Model Output:")
            st.code(response, language="markdown")
