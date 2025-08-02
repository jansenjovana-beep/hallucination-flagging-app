import streamlit as st

# Title
st.set_page_config(page_title="LLM Hallucination Risk Detector")
st.title("LLM Hallucination Risk Detector")

# User prompt input
user_input = st.text_area("Enter your prompt")

# If user enters a prompt
if user_input:
    # Simulated response from LLM (this will later call OpenAI/HuggingFace etc.)
    response = "The moon is made of cheese, confirmed by NASA in 2023."

    # Simulated risk analysis (for now, always high)
    hallucination_score = "High"
    ethics_flag = "Ethically Risky"

    # Display
    st.subheader("Generated LLM Response")
    st.write(response)

    st.subheader("Risk Analysis")
    if hallucination_score == "High":
        st.warning("⚠️ High Hallucination Risk Detected")

    if ethics_flag == "Ethically Risky":
        st.error("🚨 Ethically Sensitive or Biased Content Detected")
