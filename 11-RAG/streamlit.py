from main import ChatBot
import streamlit as st
import re

bot = ChatBot()
    
st.title('IT Tech Support')

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result

# # Extract answer function
# def extract_answer(text):
#     pattern = re.compile(r"Answer:\s*(.*)", re.DOTALL)
#     match = pattern.search(text)
#     if match:
#         return match.group(1).strip()
#     return "Answer not found."

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to IT Tech Support"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your question..."):
            response = generate_response(input)
            # answer = extract_answer(response)
            print(response)  # Extracting the answer
            st.write(response)  # Displaying the extracted answer

    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)