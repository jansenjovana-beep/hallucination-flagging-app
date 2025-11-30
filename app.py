import streamlit as st
import openai
import requests

# ---------- Config ----------
st.set_page_config(page_title="LLM Hallucination Risk Detector")
st.title("LLM Hallucination Risk Detector")

# ---------- API clients / settings ----------

# OpenAI client (GPT-3.5)
openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Hugging Face Llama endpoint
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3-8b-instruct"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}


def call_gpt35(prompt: str) -> str:
    """Call OpenAI GPT-3.5 with the user prompt."""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content


def call_llama(prompt: str) -> str:
    """Call Llama via Hugging Face Inference API."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
        },
    }
    res = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
    res.raise_for_status()
    data = res.json()


    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
       
        full_text = data[0]["generated_text"]
        # Simple heuristic: return everything after the original prompt
        if full_text.startswith(prompt):
            return full_text[len(prompt) :].strip()
        return full_text.strip()

    # Fallback for other response shapes
    return str(data)


# ---------- UI ----------

# Model selector
model_choice = st.radio(
    "Choose model:",
    ("GPT-3.5 (OpenAI)", "Llama (Hugging Face)"),
    horizontal=True,
)

# User input
user_input = st.text_area("Enter your prompt")

# ---------- Main generation + display ----------
if user_input:
    with st.spinner("Generating response..."):
        try:
            if model_choice.startswith("GPT-3.5"):
                model_used = "GPT-3.5"
                output = call_gpt35(user_input)
            else:
                model_used = "Llama"
                output = call_llama(user_input)

            st.subheader(f"Generated LLM Response ({model_used})")
            st.write(output)

          
            st.subheader("Risk Analysis (Demo)")
            st.warning("⚠️ Possible Hallucination – Not Verified (placeholder)")
            st.error("🚨 Ethical Risk Detected (Demo Placeholder)")

        except Exception as e:
            st.error(f"Error calling {model_choice}: {e}")

