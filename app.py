import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model name
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k"

# Set device (EC2 t2.xlarge has no GPU)
device = "cpu"

# Load tokenizer and model with caching
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float32  # Use float32 for CPU
    ).to(device)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("🤖 TinyLlama Chatbot")
st.write("Ask anything and get an AI-generated response!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with torch.no_grad():
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs, 
            max_length=150,  # Limit response length
            temperature=0.7,  # More natural responses
            top_p=0.9
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
