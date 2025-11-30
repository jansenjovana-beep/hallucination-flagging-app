import streamlit as st
import openai
import requests
import re
from collections import Counter

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
HF_API_URL = (
    "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"
)
HF_HEADERS = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}


# --------------------------------------------------
# Helper functions to call each model
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
    """Call Llama via Hugging Face Inference API and return the text response."""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200, "temperature": 0.7},
    }
    resp = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Typical HF Inference response: list[{"generated_text": "..."}]
    try:
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
        # Fallback – just return stringified JSON
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
    """Return True and list of hedging phrases found (case-insensitive)."""
    t = text.lower()
    hits = [p for p in HEDGING_PHRASES if p in t]
    return len(hits) > 0, hits


def detect_repetition(text: str):
    """
    Very simple repetition detector:
    split into sentences and flag if any sentence appears more than once.
    """
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]\s+", text)
        if s.strip()
    ]
    counts = Counter(sentences)
    repeated = [s for s, c in counts.items() if c > 1]
    return len(repeated) > 0, repeated


def extract_entities(text: str):
    """
    Naive 'entity' extractor:
    sequences of Capitalised Words (e.g. 'Sam Altman', 'TruthfulQA').
    Keeps everything lightweight (no spaCy download).
    """
    ents = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return set(ents)


def detect_entity_mismatch(prompt: str, response: str):
    """
    Flags when the response introduces extra named entities
    that don't appear in the prompt (possible hallucinated people/places/etc.).
    """
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
# UI
# --------------------------------------------------
st.write("")

model_choice = st.radio(
    "Choose model:",
    ("GPT-3.5 (OpenAI)", "Llama (Hugging Face)"),
    horizontal=True,
)

user_input = st.text_area("Enter your prompt", height=150)

if user_input:
    with st.spinner("Generating response..."):
        try:
            if model_choice.startswith("GPT-3.5"):
                model_used = "GPT-3.5 (OpenAI)"
                output = call_gpt35(user_input)
            else:
                model_used = "Llama (Hugging Face)"
                output = call_llama(user_input)

            st.subheader(f"Generated LLM Response – {model_used}")
            st.write(output)

            # ---------------- Hallucination heuristics ----------------
            halluc = heuristic_hallucination_score(user_input, output)

            st.subheader("Heuristic Hallucination Analysis")

            score = halluc["score"]
            label = halluc["label"]

            if score == 0:
                st.success(f"Overall heuristic risk: **{label}** (score {score}/3)")
            elif score == 1:
                st.info(f"Overall heuristic risk: **{label}** (score {score}/3)")
            else:
                st.warning(f"Overall heuristic risk: **{label}** (score {score}/3)")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(
                    "Hedging cues:",
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
                    "✅ none" if not halluc["entity_flag"] else "⚠️ possible extras",
                )

            if halluc["hedging_hits"]:
                st.caption(
                    "Hedging phrases found: "
                    + ", ".join(f"“{h}”" for h in halluc["hedging_hits"])
                )

            if halluc["extra_entities"]:
                st.caption(
                    "Extra entities in response (not in prompt): "
                    + ", ".join(halluc["extra_entities"])
                )

            if halluc["repeated_sentences"]:
                st.caption(
                    "Repeated sentences: "
                    + " | ".join(f"“{s}”" for s in halluc["repeated_sentences"])
                )

            # ---------------- Ethical risk placeholder ----------------
            st.subheader("Ethical Risk (Toxicity)")
            st.info(
                "Integration with Perspective API / Detoxify for toxicity and "
                "ethical risk scoring will be added here for Objective 3."
            )

        except Exception as e:
            st.error(f"Error while calling {model_choice}: {e}")
model_choice}: {e}")


