import streamlit as st
from google import genai
from google.genai import types

# 1. Page Configuration
st.set_page_config(page_title="Gemini Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Gemini AI Chatbot")
st.caption("A responsive chatbot powered by Streamlit and Gemini 2.5 Flash.")

# 2. Securely Initialize the Gemini Client
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
except KeyError:
    st.error("Missing API Key! Please add 'GEMINI_API_KEY' to your Streamlit secrets.")
    st.stop()

# 3. Initialize Conversation History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Render Chat History on App Rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle New User Input
if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            formatted_contents = [
                types.Content(
                    role="user" if m["role"] == "user" else "model",
                    parts=[types.Part.from_text(text=m["content"])]
                ) for m in st.session_state.messages
            ]
            
            response_stream = client.models.generate_content_stream(
                model='gemini-2.5-flash',
                contents=formatted_contents,
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful, concise AI assistant."
                )
            )
            
            for chunk in response_stream:
                if chunk.text:
                    full_response += chunk.text
                    message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
