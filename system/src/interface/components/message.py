import streamlit as st
import random
import time
from datetime import datetime

import sys
import os
from dotenv import load_dotenv

# === fix import module
load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from system.src.interface.controllers.message import MessageController

def response_generator(query: str, task: str):
    if task == "try_with_llama":
        response, docs = MessageController().get_responses(input_query=query)
        print("\n\n herre get reponse done successfull")
    elif task == "try_with_gemini":
        response, docs = MessageController().get_response_gemini(input_query=query)
        print("\n\n herre get reponse gemini done successfull")
    else:
        response, docs = "[❌ Không xác định tác vụ]", []

    st.session_state["docs"] = docs
    buffer = ""
    for char in response:
        buffer += char
        if char == '\n':
            yield buffer
            buffer = ""
    if buffer:
        yield buffer
        
    print("\n\n done reponse")

def render():
    st.subheader("📬 Messenger")

    task = st.selectbox(
        "Chọn tác vụ xử lý:",
        ["try_with_llama", "try_with_gemini"],
        index=0,
        help="Chọn phương thức phản hồi mong muốn"
    )

    st.markdown("""
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        .chat-bubble {
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 70%;
            margin-bottom: 10px;
            line-height: 1.4;
            word-wrap: break-word;
            font-size: 15px;
            display: inline-block;
        }
        .user-msg {
            align-self: flex-end;
            background-color: #0084FF;
            color: white;
            border-bottom-right-radius: 2px;
        }
        .bot-msg {
            align-self: flex-start;
            background-color: #E4E6EB;
            color: black;
            border-bottom-left-radius: 2px;
        }
        .msg-row {
            display: flex;
            align-items: flex-end;
        }
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            margin: 0 8px;
            background-color: #ccc;
            text-align: center;
            line-height: 36px;
            font-size: 20px;
        }
        .msg-row.user {
            flex-direction: row-reverse;
        }
        </style>
    """, unsafe_allow_html=True)

    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}

    if "current_chat_id" not in st.session_state:
        new_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_sessions[new_id] = []
        st.session_state.current_chat_id = new_id

    st.sidebar.title("🗂️ Chat History")

    if st.sidebar.button("➕ New Chat"):
        new_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_sessions[new_id] = []
        st.session_state.current_chat_id = new_id
        st.session_state.pop("docs", None)
        st.rerun()

    chat_ids = sorted(st.session_state.chat_sessions.keys(), reverse=True)
    selected_chat = st.sidebar.radio(
        "Select a chat:", chat_ids, index=chat_ids.index(st.session_state.current_chat_id))
    st.session_state.current_chat_id = selected_chat


    chat_history = st.session_state.chat_sessions[selected_chat]

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in chat_history:
        sender = msg["role"]
        content = msg["content"]

        if sender == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="avatar">🧑</div>
                <div class="chat-bubble user-msg">{content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row">
                <div class="avatar">🤖</div>
                <div class="chat-bubble bot-msg">{content}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if "docs" in msg and msg["docs"] is not None and len(msg["docs"]) > 0:
                with st.expander("📄 Tham khảo thêm tài liệu liên quan"):
                    for i, doc in enumerate(msg["docs"], 1):
                        st.markdown(f"""
                        <div style="font-size: 13px; line-height: 1.4">
                            <b>Tài liệu #{i}</b><br>
                            • <b>Câu hỏi:</b> {doc.get("Question", "")}<br>
                            • <b>Câu trả lời:</b> {doc.get("Answer", "")}<br>
                            • <b>Nguồn:</b> {doc.get("Source", "")}<br>
                            • <b>Chuyên mục:</b> {doc.get("Category", "")}<br>
                            • <b>Score:</b> <code>{round(doc.get("Score", 0), 4)}</code><br>
                            <hr style="margin-top: 6px; margin-bottom: 6px">
                        </div>
                        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 💡 Try with sample questions:")
    sample_questions = [
        "What is dengue fever, what are its symptoms, how can it be diagnosed, what is the treatment, and what are the complications it leaves behind?",
        "What is the capital of France?",
        "Can you explain what causes migraine, how it can be diagnosed, and what complications it leads to?",
        "What is breast cancer, what causes it, what complications can it lead to?",
        "What is depression, how should it be treated, what are the symptoms?",
        "What is kidney stone disease, is kidney stone disease dangerous, how should it be treated, are there any complications?",
        "Explain machine learning in simple terms.",
        "What is high blood pressure, what are its causes, how can it be detected early, and what are the effective treatments?",
        "What is cardiovascular disease, what are the risk factors, and how can we improve heart health?"
        " How does a blockchain work?"
    ]

    for question in sample_questions:
        if st.button(question):
            chat_history.append({"role": "user", "content": question})
            response_text = ""
            with st.spinner("Responding ..."):
                for word in response_generator(question, task):
                    response_text += word
                time.sleep(0.2)
            bot_message = {"role": "bot", "content": response_text}
            if "docs" in st.session_state:
                bot_message["docs"] = st.session_state["docs"]
            chat_history.append(bot_message)
            
            st.rerun()


    if prompt := st.chat_input("Type your message ..."):
        chat_history.append({"role": "user", "content": prompt})

        response_text = ""
        with st.spinner("Responding ..."):
            for word in response_generator(prompt, task):
                response_text += word
            time.sleep(0.2)
        
        bot_message = {"role": "bot", "content": response_text}
        if "docs" in st.session_state:
            bot_message["docs"] = st.session_state["docs"]
        chat_history.append(bot_message)

        # st.rerun()