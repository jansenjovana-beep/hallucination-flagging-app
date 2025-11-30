import streamlit as st
import openai
import requests
import re
from collections import Counter

# --------------------------------------------------
# Streamlit UI Setup
# --------------------------------------------------
st.set_page_config(page_title="LLM Hallucination Risk Detector", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------
# API clients / configuration
# --------------------------------------------------

# 1. OpenAI (GPT-3.5)
openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 2. Hugging Face router (for Llama) – optional if HF_TOKEN not set
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
hf_client = None
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

if HF_TOKEN:
    hf_client = openai.OpenAI(
        api_key=HF_TOKEN,
        base_url="https://router.huggingface.co/v1"
    )

# 3. Perspective API (toxicity)
PERSPECTIVE_API_KEY = st.secrets.get("PERSPECTIVE_API_KEY", None)
PERSPECTIVE_URL = (
    "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
)

# --------------------------------------------------
# LLM call helpers
# --------------------------------------------------

def call_gpt35(prompt: str) -> str:
    """Call OpenAI GPT-3.5 and return text."""
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def call_llama(prompt: str) -> str:
    """Call Llama via Hugging Face router and return text."""
    if hf_client is None:
        raise RuntimeError("HF_TOKEN is not configured in Streamlit secrets.")
    response = hf_client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# --------------------------------------------------
# Heuristic hallucination detection
# --------------------------------------------------

HEDGING_PHRASES = [
    "it is believed",
    "it is commonly believed",
    "some people say",
    "some claim",
    "it is thought",
    "may suggest",
    "might suggest",
    "could be",
    "possibly",
    "it seems",
    "it appears",
]


def detect_hedging(text: str):
    t = text.lower()
    hits = [p for p in HEDGING_PHRASES if p in t]
    return len(hits) > 0, hits


def detect_repetition(text: str):
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]", text)
        if s.strip()
    ]
    counts = Counter(sentences)
    repeated = [s for s, c in counts.items() if c > 1]
    return len(repeated) > 0, repeated


def extract_entities(text: str):
    # naive capitalised-word “entity” detector
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

    score = 0
    if hedging_flag:
        score += 1
    if repetition_flag:
        score += 1
    if entity_flag:
        score += 1

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
# Perspective API toxicity
# --------------------------------------------------

def perspective_toxicity(text: str):
    """Return toxicity score in [0,1] or None if not available/error."""
    if not PERSPECTIVE_API_KEY:
        return None
    try:
        payload = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}},
        }
        resp = requests.post(
            PERSPECTIVE_URL,
            params={"key": PERSPECTIVE_API_KEY},
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        score = data["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        return round(float(score), 3)
    except Exception:
        return None


# --------------------------------------------------
# UI
# --------------------------------------------------

st.write("")

model_choice = st.radio(
    "Choose model:",
    ("GPT-3.5 (OpenAI)", "Llama (Hugging Face)"),
    horizontal=True,
)

with st.form("prompt_form", clear_on_submit=False):
    user_input = st.text_area("Enter your prompt", height=150)
    submitted = st.form_submit_button("Submit")

if submitted and user_input:
    with st.spinner("Generating response..."):
        try:
            if model_choice.startswith("GPT-3.5"):
                model_used = "GPT-3.5 (OpenAI)"
                output = call_gpt35(user_input)
            else:
                model_used = "Llama (Hugging Face)"
                output = call_llama(user_input)

            # ---------------- Generated response ----------------
            st.subheader(f"Generated Response – {model_used}")
            st.write(output)

            # ---------------- Hallucination heuristics ----------------
            st.subheader("Hallucination Heuristics")
            halluc = heuristic_hallucination_score(user_input, output)

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

            # ---------------- Ethical risk (toxicity) ----------------
            st.subheader("Ethical Risk (Toxicity)")
            tox = perspective_toxicity(output)
            if tox is None:
                st.info(
                    "Perspective API score: N/A "
                    "(no key configured or request failed)."
                )
            else:
                st.write(f"Perspective API toxicity score: **{tox}**")

            # ---------------- Final combined risk ----------------
            st.subheader("Final Combined Risk")

            # simple combination: hallucination score + toxicity bonus
            tox_bonus = 1 if (tox is not None and tox >= 0.6) else 0
            combined = score + tox_bonus

            if combined <= 1:
                st.success("🟢 Overall: Low Risk")
            elif combined == 2:
                st.warning("🟡 Overall: Moderate Risk")
            else:
                st.error("🔴 Overall: High Risk")

        except Exception as e:
            st.error(f"Error while calling {model_choice}: {e}")


