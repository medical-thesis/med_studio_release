import streamlit as st
import pandas as pd
from system.src.interface.controllers.translate import TranslateController

languages = ["English", "Vietnamese"]

def render():
    st.subheader("🌐 Medical translate")

    model_names = ["vietai", "vinai", "cohere", "opus"]
    selected_model = st.session_state.get("selected_model", "vietai")

    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            if st.button(model_names[i].capitalize()):
                st.session_state["selected_model"] = model_names[i]
                selected_model = model_names[i]

    st.markdown(f"**Selected model:** `{selected_model}`")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Text")
        src_lang = st.selectbox("Select language:", languages, index=0)
        input_text = st.text_area("Enter text to translate:", height=300, key="input_text")
        submit = st.button("Translate")

    with col2:
        st.subheader("Output Text")
        dest_lang = st.selectbox("Select language:", ["Vietnamese" if src_lang == "English" else "English"], index=0, disabled=True)

        if submit and input_text:
            result = TranslateController().translate(
                text=input_text,
                source_lang="en" if src_lang == "English" else "vi",
                target_lang="vi" if src_lang == "English" else "en",
                model=selected_model
            )
        else:
            result = ""

        st.text_area("Translated text will appear here:", value=result, height=300, disabled=True)

    data = {
        "col1": ["What are the symptoms of Glaucoma?", 
                 "How are you today?", 
                 "Nghiên cứu đặc điểm lâm sàng, cận lâm sàng bệnh nhân viêm tai ứ dịch trên viêm V.A tại Khoa Tai mũi họng - Bệnh viện Trung ương Thái Nguyên", 
                 "Hi, I think my child might have bronchiolitis. Can you provide more information on this condition?", 
                 "Hình ảnh nội soi màng nhĩ điển hình là màng nhĩ lõm, màu vàng hoặc có bóng khí."],
        # "col2": ["bạn có khỏe không", "f", "g", "h"],
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
