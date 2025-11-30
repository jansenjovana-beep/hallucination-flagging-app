import streamlit as st
import openai
import requests
import re
from collections import Counter
from typing import Optional

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(page_title="LLM Hallucination Risk Detector", layout="wide")
st.title("LLM Hallucination Risk Detector")

# --------------------------------------------------
# API clients / config
# --------------------------------------------------
# OpenAI (GPT-3.5)
openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Hugging Face (Llama)
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

# Perspective API (for toxicity)
PERSPECTIVE_API_KEY = st.secrets.get("PERSPECTIVE_API_KEY", None)
PERSPECTIVE_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

# --------------------------------------------------
# Model calling functions
# --------------------------------------------------
def call_gpt35(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def call_llama(prompt: str) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7},
    }
    resp = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list) and len(data) > 0:
        if "generated_text" in data[0]:
            return data[0]["generated_text"].strip()

    return str(data)

# --------------------------------------------------
# Hallucination heuristics
# --------------------------------------------------
HEDGING_PHRASES = [
    "it is believed that",
    "some people say",
    "it seems that",
    "it appears that",
    "possibly",
    "might",
    "may suggest",
    "could be",
]


def detect_hedging(text):
    t = text.lower()
    hits = [p for p in HEDGING_PHRASES if p in t]
    return len(hits) > 0, hits


def detect_repetition(text):
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]\s+", text)
        if s.strip()
    ]
    counts = Counter(sentences)
    repeated = [s for s, c in counts.items() if c > 1]
    return len(repeated) > 0, repeated


def extract_entities(text):
    ents = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return set(ents)


def detect_entity_mismatch(prompt, response):
    p_ents = extract_entities(prompt)
    r_ents = extract_entities(response)
    extra = r_ents - p_ents
    return len(extra) > 0, extra, p_ents, r_ents


def heuristic_hallucination_score(prompt, response):
    hedging_flag, hedging_hits = detect_hedging(response)
    repetition_flag, repeated = detect_repetition(response)
    entity_flag, extra_ents, p_ents, r_ents = detect_entity_mismatch(prompt, response)

    score = sum([hedging_flag, repetition_flag, entity_flag])

    if score == 0:
        label = "Low (likely grounded)"
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
# Perspective API Toxicity scoring
# --------------------------------------------------
def score_toxicity_perspective(text) -> Optional[float]:
    if not PERSPECTIVE_API_KEY:
        return None

    payload = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {"TOXICITY": {}},
    }

    try:
        resp = requests.post(
            PERSPECTIVE_URL,
            params={"key": PERSPECTIVE_API_KEY},
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        score = data["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        return float(score)
    except Exception:
        return None

# --------------------------------------------------
# Combine hallucination + toxicity into final risk
# --------------------------------------------------
def overall_risk(halluc_score: int, tox_score: Optional[float]):
    if tox_score is not None:
        if tox_score >= 0.7:
            return "High"
        if tox_score >= 0.4:
            return "Moderate"

    if halluc_score >= 2:
        return "High"
    if halluc_score == 1:
        return "Moderate"
    return "Low"

# --------------------------------------------------
# UI
# --------------------------------------------------
model_choice = st.radio(
    "Choose model:",
    ("GPT-3.5 (OpenAI)", "Llama (Hugging Face)"),
    horizontal=True,
)

user_input = st.text_area("Enter your prompt", height=160)

if user_input:
    with st.spinner("Generating response..."):
        try:
            output = call_gpt35(user_input) if model_choice.startswith("GPT") else call_llama(user_input)

            st.subheader(f"Generated Response – {model_choice}")
            st.write(output)

            # ---------------- Hallucination heuristics ----------------
            halluc = heuristic_hallucination_score(user_input, output)
            h_score = halluc["score"]

            st.subheader("Hallucination Heuristics")
            st.write(f"**Heuristic score:** {h_score}/3 — **{halluc['label']}**")

            # ---------------- Toxicity scoring ----------------
            st.subheader("Ethical Risk (Toxicity)")
            tox_score = score_toxicity_perspective(output)

            st.write(f"Perspective API toxicity score: **{tox_score if tox_score is not None else 'N/A'}**")

            final_risk = overall_risk(h_score, tox_score)

            st.subheader("Final Combined Risk")

            if final_risk == "High":
                st.error("🚨 **High Risk**")
            elif final_risk == "Moderate":
                st.warning("⚠️ **Moderate Risk**")
            else:
                st.success("🟢 **Low Risk**")

        except Exception as e:
            st.error(f"Error: {e}")
