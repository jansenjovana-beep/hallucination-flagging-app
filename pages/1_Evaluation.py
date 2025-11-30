import streamlit as st
import pandas as pd

st.title("Evaluation: Hallucination & Ethical Risk Detection")

# -----------------------------------------------------
# 1. LOAD DATASET
# -----------------------------------------------------
DATA_PATH = "data/eval_prompts.csv"

try:
    df = pd.read_csv(DATA_PATH)
    st.success(f"Loaded dataset with {len(df)} rows.")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Could not load `{DATA_PATH}`.\nError: {e}")
    st.stop()

st.markdown("---")

# -----------------------------------------------------
# 2. MODEL SELECTOR
# -----------------------------------------------------
model_choice = st.selectbox(
    "Choose model:",
    ["GPT-3.5 (OpenAI)", "Llama (Hugging Face)"]
)

# -----------------------------------------------------
# 3. RUN BUTTON (THE ONE YOU COULDN’T SEE)
# -----------------------------------------------------
run_eval = st.button("Run evaluation on dataset")

# This is where the button will appear.
#             ↑
# If you do NOT see this button in the UI,
# the file is not loading correctly.

# -----------------------------------------------------
# 4. HANDLER: When user clicks button...
# -----------------------------------------------------
if run_eval:
    st.info("Running evaluation... (placeholder)")
    st.write("The full evaluation logic will go here next.")
