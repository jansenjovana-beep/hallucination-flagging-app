import streamlit as st
import openai

# App config
st.set_page_config(page_title="LLM Hallucination Risk Detector")
st.title("LLM Hallucination Risk Detector")

# Load OpenAI key securely from Streamlit Secrets
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# User input
user_input = st.text_area("Enter your prompt")

# Send to OpenAI and display result
if user_input:
    with st.spinner("Generating response..."):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}],
                temperature=0.7,
                max_tokens=200
            )
            output = response.choices[0].message.content

            st.subheader("Generated LLM Response")
            st.write(output)

            st.subheader("Risk Analysis (Demo)")
            st.warning("⚠️ Possible Hallucination – Not Verified")
            st.error("🚨 Ethical Risk Detected (Demo Placeholder)")

        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
