import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Hugging Face chat model
MODEL_NAME = "microsoft/phi-2"

# Use CPU and optimize memory
device = "cpu"  # t2.xlarge has no GPU
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch_dtype,  # Use appropriate precision
    device_map="auto"  # Automatically place on CPU
)

# Streamlit UI
st.title("💬 Chatbot using Hugging Face")
st.write("Ask anything and get a meaningful AI-generated response!")

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
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=100)  # Limit token length
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display AI response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
