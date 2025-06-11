import streamlit as st

def render():
    st.subheader("🏠 Home")
    st.markdown("""
    <div style='text-align: center; font-size: 18px;'>
        Welcome to the custom multi-page app.<br>
        Use the sidebar to navigate.
    </div>
    """, unsafe_allow_html=True)
