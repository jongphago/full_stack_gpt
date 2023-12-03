import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

with st.chat_message("human"):
    st.write("Hello, I am a human!")

with st.chat_message("ai"):
    st.write("Hello, I am an AI!")

st.chat_input("Say something to me!")

with st.status("Embedding file...", expanded=True) as status:
    time.sleep(3)
    st.write("Getting the file...")
    time.sleep(3)
    st.write("Embeddingthe file...")
    time.sleep(3)
    st.write("Caching the file...")
    status.update(label="Error", state="error")
