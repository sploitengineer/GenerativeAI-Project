import streamlit as st
from langchain_ollama import ChatOllama


st.set_page_config(page_title="Chatbot", layout="wide")
st.title("Chatbot")

##initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]

##render chat message
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**User:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

st.divider()

user_input = st.chat_input("Type your message here:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    #clearing input box
    st.session_state.input = ""

    llm = ChatOllama(model="llama3:8b")

    response = llm.invoke(
        [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
    )
    assistant_reply = response.content.strip()

    ##append assistant reply
    st.session_state.messages.append({"role":"assistant", "content": assistant_reply})

    ##rerun to display updated chat
    st.rerun()