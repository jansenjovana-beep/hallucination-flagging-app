import streamlit as st
import openai
import requests
import re
from collections import Counter
from detoxify import Detoxify

# --------------------------------------------------
# Streamlit UI Setup
# --------------------------------------------------
st.set_page_config(page_title="LLM Hallucination Risk Detector", layout="wide")
st.title("LLM Hallucination Risk Detector")

# --------------------------------------------------
# API Clients
# --------------------------------------------------

# 1. OpenAI (GPT-3.5)
openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. HuggingFace Router (OpenAI-compatible)
hf_client = openai.OpenAI(
    api_key=st.secrets["HF_TOKEN"],
    base_url="https://router.huggingface.co/v1"
)
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# 3. Perspective API
PERSPECTIVE_KEY = st.secrets["PERSPECTIVE_API_KEY"]
PERSPECTIVE_URL = (
    "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
)

# --------------------------------------------------
# MODEL CALLS
# --------------------------------------------------

def call_gpt35(prompt: str) -> str:
    """Call GPT-3.5 Turbo."""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


def call_llama(prompt: str) -> str:
    """Call Llama via HuggingFace router."""
    response = hf_client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------
# HEURISTIC HALLUCINATION DETECTION
# --------------------------------------------------

HEDGING_PHRASES = [
    "it is believed", "some people say", "it is thought",
    "suggests that", "may indicate", "possibly", "could be",
    "it appears", "it seems"
]

def detect_hedging(text: str):
    t = text.lower()
    hits = [p for p in HEDGING_PHRASES if p in t]
    return len(hits) > 0, hits

def detect_repetition(text: str):
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    counts = Counter(sentences)
    repeated = [s for s, c in counts.items() if c > 1]
    return len(repeated) > 0, repeated

def extract_entities(text: str):
    ents = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return set(ents)

def detect_entity_mismatch(prompt: str, response: str):
    p_ents = extract_entities(prompt)
    r_ents = extract_entities(response)
    extra = r_ents - p_ents
    return len(extra) > 0, extra, p_ents, r_ents

def heuristic_hallucination_score(prompt: str, response: str):
    hedging_flag, hedging_hits = detect_hedging(response)
    repetition_flag, repeated = detect_repetition(response)
    entity_flag, extra_ents, p_ents, r_ents = detect_entity_mismatch(prompt, response)

    score = hedging_flag + repetition_flag + entity_flag

    if score == 0:
        label = "Low"
    elif score == 1:
        label = "Moderate"
    else:
        label = "High"

    return {
        "score": score,
        "label": label,
        "hedging_flag": hedging_flag,
        "repetition_flag": repetition_flag,
        "entity_flag": entity_flag,
        "hedging_hits": hedging_hits,
        "repeated_sentences": repeated,
        "extra_entities": list(extra_ents),
    }

# --------------------------------------------------
# PERSPECTIVE API – Toxicity
# --------------------------------------------------

def get_perspective_score(text: str):
    try:
        data = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}}
        }
        resp = requests.post(
            PERSPECTIVE_URL,
            params={"key": PERSPECTIVE_KEY},
            json=data,
            timeout=10
        )
        resp.raise_for_status()
        toxicity = resp.json()["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        return round(toxicity, 3)
    except:
        return None

# --------------------------------------------------
# DETOXIFY – Toxicity Model
# --------------------------------------------------

def detoxify_score(text: str):
    try:
        model = Detoxify("original")
        out = model.predict(text)
        return round(out["toxicity"], 3)
    except:
        return None

# --------------------------------------------------
# UI – Model Selector
# --------------------------------------------------

model_choice = st.radio(
    "Choose model:",
    ("GPT-3.5 (OpenAI)", "Llama (Hugging Face)"),
    horizontal=True
)

user_input = st.text_area("Enter your prompt", height=150)

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------

if user_input:
    with st.spinner("Generating response..."):
        try:
            if model_choice.startswith("GPT"):
                model_used = "GPT-3.5 (OpenAI)"
                output = call_gpt35(user_input)
            else:
                model_used = "Llama (Hugging Face)"
                output = call_llama(user_input)

            st.subheader(f"Generated Response – {model_used}")
            st.write(output)

            # ------------------ Heuristics ------------------
            halluc = heuristic_hallucination_score(user_input, output)

            st.subheader("Hallucination Heuristics")
            st.write(f"Heuristic score: {halluc['score']}/3 — **{halluc['label']}**")

            if halluc["hedging_hits"]:
                st.caption("Hedging phrases: " + ", ".join(halluc["hedging_hits"]))

            if halluc["repeated_sentences"]:
                st.caption(
                    "Repeated sentences: " +
                    " | ".join(halluc["repeated_sentences"])
                )

            if halluc["extra_entities"]:
                st.caption(
                    "Extra entities not in prompt: " +
                    ", ".join(halluc["extra_entities"])
                )

            # ------------------ Ethical Risk ------------------
            st.subheader("Ethical Risk (Toxicity)")

            persp = get_perspective_score(output)
            detox = detoxify_score(output)

            st.write(f"Perspective API toxicity score: {persp}")
            st.write(f"Detoxify toxicity score: {detox}")

            # ------------------ Final Combined ------------------
            st.subheader("Final Combined Risk")

            combined_score = halluc["score"] + (
                1 if (persp and persp > 0.6) else 0
            ) + (
                1 if (detox and detox > 0.6) else 0
            )

            if combined_score <= 1:
                st.success("🟢 Low Risk")
            elif combined_score == 2:
                st.warning("🟡 Moderate Risk")
            else:
                st.error("🔴 High Risk")

        except Exception as e:
            st.error(f"Error while calling model: {e}")
