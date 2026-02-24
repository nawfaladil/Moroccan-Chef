import requests
import streamlit as st

st.text_input("ask about a moroccan recipe!", key="chat_box")

response = requests.get(f"http://127.0.0.1:8000/generate/{st.session_state.chat_box}")

st.write(response.content)
