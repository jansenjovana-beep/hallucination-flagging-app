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

# Hugging Face (Llama via Router)
HF_API_URL = "https://router.huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

# Perspective API
PERSPECTIVE_API_KEY = st.secrets.get("PERSPECTIVE_API_KEY", None)
PERSPECTIVE_URL = (
    "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    if PERSPECTIVE_API_KEY
    else None
)

# --------------------------------------------------
# Model call helpers
# --------------------------------------------------


def call_gpt35(prompt: str) -> str:
    """Call OpenAI GPT-3.5 and return the text response."""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def call_llama(prompt: str) -> str:
    """Call Llama via Hugging Face Router and return the text response."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7},
    }
    resp = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Typical HF inference output: list[{"generated_text": "..."}]
    try:
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
        return str(data)
    except Exception:
        return str(data)


# --------------------------------------------------
# Heuristic hallucination detection
# --------------------------------------------------

HEDGING_PHRASES = [
    "it is believed that",
    "it is commonly believed",
    "some people say",
    "some claim",
    "it is thought that",
    "may suggest",
    "might suggest",
    "could be",
    "possibly",
    "it seems that",
    "it appears that",
]


def detect_hedging(text: str):
    """Return (flag, list_of_hits) for hedging phrases."""
    t = text.lower()
    hits = [p for p in HEDGING_PHRASES if p in t]
    return len(hits) > 0, hits


def detect_repetition(text: str):
    """Very simple repetition detector based on duplicate sentences."""
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    counts = Counter(sentences)
    repeated = [s for s, c in counts.items() if c > 1]
    return len(repeated) > 0, repeated


def extract_entities(text: str):
    """
    Naive 'entity' extractor:
    Sequences of Capitalised Words (e.g. 'Sam Altman', 'TruthfulQA').
    """
    ents = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return set(ents)


def detect_entity_mismatch(prompt: str, response: str):
    """Extra entities in the response that were not in the prompt."""
    p_ents = extract_entities(prompt)
    r_ents = extract_entities(response)
    extra = r_ents - p_ents
    return len(extra) > 0, extra, p_ents, r_ents


def heuristic_hallucination_score(prompt: str, response: str):
    """
    Combine all three heuristics into a 0–3 score and a label.
    0 = Low, 1 = Moderate, 2–3 = High.
    """
    hedging_flag, hedging_hits = detect_hedging(response)
    repetition_flag, repeated = detect_repetition(response)
    entity_flag, extra_ents, p_ents, r_ents = detect_entity_mismatch(prompt, response)

    score = 0
    if hedging_flag:
        score += 1
    if repetition_flag:
        score += 1
    if entity_flag:
        score += 1

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
        "prompt_entities": list(p_ents),
        "response_entities": list(r_ents),
    }


# --------------------------------------------------
# Perspective API toxicity scoring
# --------------------------------------------------


def get_perspective_toxicity(text: str) -> Optional[float]:
    """Return toxicity score in [0,1] or None if not available."""
    if not PERSPECTIVE_API_KEY or not PERSPECTIVE_URL:
        return None

    try:
        payload = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}},
        }
        params = {"key": PERSPECTIVE_API_KEY}
        resp = requests.post(PERSPECTIVE_URL, params=params, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        score = data["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        return float(score)
    except Exception:
        return None


# --------------------------------------------------
# Risk combination logic
# --------------------------------------------------


def combined_risk(halluc_score: int, tox_score: Optional[float]):
    """
    Map hallucination + toxicity into a final label and level.
    level is one of 'low', 'moderate', 'high'.
    """
    tox = tox_score if tox_score is not None else 0.0

    if tox >= 0.7 or halluc_score >= 3:
        return "High Risk", "high"
    elif tox >= 0.4 or halluc_score >= 2:
        return "Moderate Risk", "moderate"
    else:
        return "Low Risk", "low"


# --------------------------------------------------
# UI
# --------------------------------------------------

st.write("")

model_choice = st.radio(
    "Choose model:",
    ("GPT-3.5 (OpenAI)", "Llama (Hugging Face)"),
    horizontal=True,
)

# Single-line input: ENTER submits (no button required)
prompt = st.text_input("Enter your prompt (press Enter to submit)", value="", key="prompt")

if prompt.strip():
    with st.spinner("Generating response..."):
        try:
            # ---------------- Model call ----------------
            if model_choice.startswith("GPT-3.5"):
                model_used = "GPT-3.5 (OpenAI)"
                output = call_gpt35(prompt)
            else:
                model_used = "Llama (Hugging Face)"
                output = call_llama(prompt)

            # ---------------- Display response ----------------
            st.subheader(f"Generated Response – {model_used}")
            st.write(output)

            # ---------------- Hallucination heuristics ----------------
            st.subheader("Hallucination Heuristics")
            halluc = heuristic_hallucination_score(prompt, output)
            score = halluc["score"]
            label = halluc["label"]

            st.write(f"Heuristic score: {score}/3 — **{label}**")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(
                    "Hedging:",
                    "✅ none" if not halluc["hedging_flag"] else "⚠️ detected",
                )
            with col2:
                st.write(
                    "Repetition:",
                    "✅ none" if not halluc["repetition_flag"] else "⚠️ detected",
                )
            with col3:
                st.write(
                    "Entity mismatch:",
                    "✅ none" if not halluc["entity_flag"] else "⚠️ extras in answer",
                )

            if halluc["hedging_hits"]:
                st.caption(
                    "Hedging phrases: "
                    + ", ".join(f"“{h}”" for h in halluc["hedging_hits"])
                )

            if halluc["extra_entities"]:
                st.caption(
                    "Extra entities (not in prompt): "
                    + ", ".join(halluc["extra_entities"])
                )

            if halluc["repeated_sentences"]:
                st.caption(
                    "Repeated sentences: "
                    + " | ".join(f"“{s}”" for s in halluc["repeated_sentences"])
                )

            # ---------------- Ethical risk via Perspective ----------------
            st.subheader("Ethical Risk (Toxicity)")

            tox_score = get_perspective_toxicity(output)
            if tox_score is None:
                st.info("Perspective API toxicity score: N/A (no key configured or request failed).")
            else:
                st.write(f"Perspective API toxicity score: **{tox_score:.3f}**")

            # ---------------- Combined risk ----------------
            st.subheader("Final Combined Risk")
            final_label, level = combined_risk(score, tox_score)

            if level == "low":
                st.success(f"🟢 Overall: {final_label}")
            elif level == "moderate":
                st.warning(f"🟡 Overall: {final_label}")
            else:
                st.error(f"🔴 Overall: {final_label}")

        except Exception as e:
            st.error(f"Error while calling {model_choice}: {e}")
