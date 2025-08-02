import streamlit as st

st.set_page_config(page_title="Ethical Hallucination Flagger")

st.title("LLM Hallucination Risk Detector")

user_input = st.text_area("Enter your prompt")

if user_input:
    # In the real app, you'd query the LLM here
    fake_response = "The moon is made of cheese. NASA confirms it."

    st.subheader("LLM Response")
    st.write(fake_response)

    # Simulated hallucination and ethics score
    st.subheader("Risk Analysis")
    st.warning("⚠️ High Hallucination Risk")
    st.error("🚨 Ethically Sensitive Content Detected")
