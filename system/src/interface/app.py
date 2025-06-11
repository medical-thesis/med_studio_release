import streamlit as st

import warnings
warnings.filterwarnings('ignore')

import sys
import os
from dotenv import load_dotenv

# === fix import module
load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.src.interface.components import chat_with_image, home, medical_vqa, query, message, translate, drug_discovery, medical_search, message_base
# import home
# import message
# import query

st.set_page_config(page_title="Custom Multi-page App", layout="wide")

# Custom sidebar navigation
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "💬 Message", "💬 Basic message", "🔍 Medical retrieval", "🌐 Medical translate", "💬 Message with image", "🎰 Medical VQA", "💊 Drug discovery", "🩺 Medical search"])

# Function to load and apply CSS
def load_styles(styles_path):
    with open(styles_path) as styles:
        st.markdown(
            f'<style>{styles.read()}</style>', 
            unsafe_allow_html=True
        )

# Load and apply the CSS file at the start of the app
load_styles('system/src/interface/app.styles.css')

# Render trang theo lựa chọn
if page == "🏠 Home":
    home.render()
elif page == "💬 Message":
    message.render()
elif page == "💬 Basic message":
    message_base.render()
elif page == "🔍 Medical retrieval":
    query.render()
elif page == "🌐 Medical translate":
    translate.render()
elif page == "💬 Message with image":
    chat_with_image.render()
elif page == "🎰 Medical VQA":
    medical_vqa.render()
elif page == "💊 Drug discovery":
    drug_discovery.render()
elif page == "🩺 Medical search":
    medical_search.render()