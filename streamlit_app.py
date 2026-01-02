import streamlit as st
st.set_page_config(page_title="Sign ↔ Speech MVP", layout="wide")
st.title("Sign-Language-to-Speech • MVP")
st.markdown("- Left: run the camera script\n- Right: shows last detected word")
last_word = st.session_state.get("last_word", "—")
col1, col2 = st.columns(2)
with col1: 
    st.info("Run `python3 src/mediapipe_cam.py` in a terminal to show the camera window.")
with col2:
    st.subheader("Detected")
    st.markdown(f"<h1 style='font-size:64px;margin-top:0'>{last_word}</h1>", unsafe_allow_html=True)
