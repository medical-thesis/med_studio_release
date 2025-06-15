import streamlit as st
import pandas as pd

def render():
    st.subheader("🏠 Home")
    st.subheader("MedStudio | Comprehensive Intelligent Medical Platform")

    with open("system/src/interface/draft_manuscript_v2.pdf", "rb") as f:
        pdf_data = f.read()

    st.download_button(
        label="📄 View draft manuscript",
        data=pdf_data,
        file_name="system/src/interface/draft_manuscript_v2.pdf",
        mime="application/pdf"
    )

    st.header("💈 Overview about MedStudio")
    st.subheader("📬🧬 What is MedStudio?")
    st.markdown("""
    In this research, we developed **MedStudio**, a comprehensive intelligent medical platform that integrates advanced deep learning techniques. At its core, the system leverages the power of:
    - Large language models (LLMs), and
    - Neural network-based machine translation models

    to tackle natural language processing tasks in the medical domain such as:
    - Medical information retrieval  
    - Medical question answering  
    - Specialized medical machine translation
    """)

    st.subheader("🧠 Vision-Language Integration")
    st.markdown("""
    We also developed two modules using Vision-Language Models to address multimodal medical tasks:
    - Medical visual question answering  
    - Medical image classification and lesion segmentation, integrated with disease information retrieval and question answering
    """)

    st.subheader("🌐 Multilingual Support")
    st.markdown("""
    MedStudio is designed to support specialized languages and can be fine-tuned for different language domains.

    In this research, the system focuses on **Vietnamese-English machine translation**, aimed at building a smart medical platform for Vietnamese users with strong Vietnamese language support.
    """)

    st.subheader("💊 VietMeD: Vietnamese Pharmaceuticals Dataset")
    st.markdown("""
    As part of this research, we collected, preprocessed, and manually constructed a high-quality dataset:

    **VietMeD: Vietnamese Pharmaceuticals Dataset**
    - Includes standardized information about ~200 drugs and supplements  
    - Built from trusted Vietnamese pharmacopoeia sources  
    - Serves as a foundation for a larger Vietnamese medical dataset in the future
    """)

    st.header("📌 Proposed Method")
    st.subheader("🧱 System Architecture (Main Modules)")

    st.markdown("""
    MedStudio adopts a **multi-module, multi-modal architecture** composed of the following:

    🔹 **Main Modules:**
    - Medical Information Retrieval Module  
    - Medical Machine Translation Module  
    - Medical Question Answering Module, including:
        - Medical Data Collection and Preprocessing Module  
        - Data Indexing and Storage Module  
        - Medical Machine Translation Module  
        - Query Processing and Preprocessing Module  
        - Medical Information Retrieval Module  
        - Answer Generation Module (from Data and Query Context)
    """)

    st.subheader("🧱 System Architecture (Vision-Language Modules)")
    st.markdown("""
    🔹 **Vision-Language Modules:**
    - Lesion Segmentation and Medical Question Answering Module, including:
        - Medical Image Classification Module  
        - Lesion Segmentation in Medical Images Module  
        - Medical Visual Question Answering Module
    """)

    st.markdown("### 🔹 Pharmacological Q&A")
    st.markdown("""
    - Drug Information Retrieval and Q&A Based on the Vietnamese Pharmacopoeia
    """)

    st.markdown("""
    📮 **MedStudio aims to be a comprehensive intelligent medical platform tailored for Vietnamese users, powered by state-of-the-art AI technologies.**
    """)


    st.header("🧪 MedStudio Experimental Results")

    tab1, tab2, tab3, tab4 = st.tabs([
        "QA Module", 
        "Machine Translation", 
        "Medical Image Classification", 
        "Lesion Segmentation"
    ])

    with tab1:
        st.subheader("Medical Question Answering Module")

        with st.expander("🔍 Elasticsearch + LLaMA 3.2"):
            st.markdown("**Average Evaluation Metrics**")
            st.metric("Context Relevance", "0.778")
            st.metric("Answer Relevance", "0.775")
            st.metric("Diversity", "0.736")
            st.metric("Answer Length", "114.4")
            st.metric("Ground Truth Length", "212.86")

        with st.expander("🧠 Qdrant + Cohere"):
            st.markdown("**Average Evaluation Metrics**")
            st.metric("Context Relevance", "0.780")
            st.metric("Answer Relevance", "0.730")
            st.metric("Diversity", "0.729")
            st.metric("Answer Length", "103.14")
            st.metric("Ground Truth Length", "212.86")


    with tab2:
        st.subheader("Medical Machine Translation Module")

        data_mt = pd.DataFrame({
            "Model": [
                "VietAI/envit5-translation",
                "our-team/envit5-translation-fine-tuning/checkpoint-2500",
                "our-team/envit5-translation-fine-tuning/checkpoint-75000",
                "facebook/mbart-large-50-many-to-many-mmt",
                "our-team/mbart-large-50-many-to-many-mmt-finetuned-vi-to-en",
                "our-team/mbart-large-50-many-to-many-mmt-finetuned-vi-to-en",
            ],
            "Number of Q&A pairs": [250, 250, 250, 20, 20, 250],
            "SacreBLEU Score": [31.26, 42.67, 49.07, 27.87, 71.41, 71.28],
            "Number of test steps": [63, 63, 63, 5, 5, 63]
        })
        st.dataframe(data_mt, use_container_width=True)


    with tab3:
        st.subheader("Medical Image Classification Module")

        st.markdown("#### 📊 Preprocessed Balanced Dataset")
        dataset = pd.DataFrame({
            "Class": ["MEL", "BKL", "NV", "BCC", "AKIEC", "VASC", "DF"],
            "Train": [890, 879, 800, 411, 262, 114, 92],
            "Validation": [223, 220, 200, 103, 65, 28, 23]
        })
        st.dataframe(dataset, use_container_width=True)

        st.markdown("#### 📈 Batch Prediction Accuracy (Validation)")
        batch_accuracy = pd.DataFrame({
            "Accuracy": ["82.00%", "78.00%", "77.50%", "77.89%"],
            "Correct / Total": ["41/50", "390/500", "310/400", "296/380"]
        })
        st.dataframe(batch_accuracy, use_container_width=True)


    with tab4:
        st.subheader("Medical Lesion Segmentation Module")

        st.markdown("#### 🔧 Training Metrics Summary")
        st.metric("Min Train Loss", "0.0441")
        st.metric("Max Train IoU", "0.9515")
        st.metric("Max Train Dice", "0.8872")
        st.metric("Training Time", "705 minutes")

        st.markdown("#### 📉 Detailed Epoch Training Log")
        epoch_data = pd.DataFrame({
            "Epoch": list(range(1, 23)),
            "Train Loss": [
                0.4715, 0.2918, 0.2207, 0.1773, 0.1519, 0.1275, 0.1098, 0.0942, 0.0849, 0.0769,
                0.0747, 0.0694, 0.0629, 0.0603, 0.0560, 0.0540, 0.0514, 0.0481, 0.0481, 0.0463,
                0.0442, 0.0441
            ],
            "Train IoU": [
                0.5861, 0.7104, 0.7737, 0.8147, 0.8396, 0.8635, 0.8814, 0.8968, 0.9070, 0.9158,
                0.9187, 0.9240, 0.9307, 0.9339, 0.9382, 0.9409, 0.9433, 0.9468, 0.9472, 0.9491,
                0.9514, 0.9515
            ],
            "Train Dice": [
                0.8683, 0.8567, 0.8871, 0.8877, 0.8806, 0.8837, 0.8862, 0.8831, 0.8869, 0.8791,
                0.8849, 0.8868, 0.8832, 0.8870, 0.8874, 0.8869, 0.8821, 0.8838, 0.8861, 0.8871,
                0.8872, 0.8859
            ]
        })
        st.line_chart(epoch_data.set_index("Epoch"))
