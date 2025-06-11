import numpy as np
import torch
from PIL import Image
import streamlit as st

from datasets import load_dataset
from transformers import BlipForQuestionAnswering, BlipProcessor


# from model_utils import load_model
# from data_utils import load_pathvqa_dataset, preprocess_image
# from med_vqa.inference import predict

def load_model_v1(
        model_path='E:/source_code/nlp/med_studio_core/src/blip-vqa-finetuned', 
        processor_path='E:/source_code/nlp/med_studio_core/src/blip-vqa-finetuned', 
        device='cpu'
        ):
    model = BlipForQuestionAnswering.from_pretrained(model_path)
    processor = BlipProcessor.from_pretrained(processor_path)
    model.to(device)
    return model, processor

def load_model_v2(
        model_path='E:/trieutm/med_studio/system/src/interface/finetuned_models/med_vqa/model',
        processor_path='E:/trieutm/med_studio/system/src/interface/finetuned_models/med_vqa/processor',
        device='cpu'
        ):
    model = BlipForQuestionAnswering.from_pretrained(pretrained_model_name_or_path=model_path)
    processor = BlipProcessor.from_pretrained(pretrained_model_name_or_path=processor_path)
    model.to(device)
    return model, processor


def load_pathvqa_dataset(token):
    dataset = load_dataset("flaviagiammarino/path-vqa", token=token)
    return dataset


def preprocess_image(image, image_processor):
    image = image.convert("RGB")
    return image_processor(image, return_tensors="pt")["pixel_values"][0]


def predict(image, question, model, processor, device='cpu'):
    model.eval()
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs)
        answer = processor.tokenizer.decode(
            out_ids[0], skip_special_tokens=True)
    return answer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, processor = load_model_v2(device=device)
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="system/src/core/config/.env")
token = os.getenv("HF_AUTH_TOKEN")
dataset = load_pathvqa_dataset(token)
examples = dataset["train"]

# st.set_page_config(page_title="BLIP VQA Demo")
def render():
    st.subheader("🎰 MedVQA Studio: Medical visual question answering")

    model_option = st.selectbox(
        "Lựa chọn config của fine-tuned model:",
        (
            # "Fine-tuned on 1,000 samples and 20 epochs - Faster inference but less accurate",
            "Fine-tuned on 10,000 samples and 50 epochs - Slower inference but more accurate"
        )
    )

    if model_option.startswith("Fine-tuned on 10,000"):
        model, processor = load_model_v2(device=device)
    else:
        model, processor = load_model_v2(device=device)    

    option = st.selectbox("Chọn ảnh mẫu từ kho dữ liệu:", list(range(10)))
    sample = examples[option]
    image = sample["image"].convert("RGB")
    question = sample["question"]
    true_answer = sample["answer"]

    st.image(image, caption="Ảnh từ dataset", width=450)
    st.write(f"**Câu hỏi mẫu từ kho dữ liệu**: {question}")
    if st.button("Dự đoán với câu hỏi mẫu"):
        pixel_values = preprocess_image(image, processor).unsqueeze(0).to(device)
        inputs = processor(text=question, return_tensors="pt").to(device)
        inputs["pixel_values"] = pixel_values
        pred = predict(image, question, model, processor, device=device)
        
        st.markdown("---")
        st.error(f"📌 Câu trả lời dự đoán (predicted): {pred}")
        st.info(f"✅ Câu trả lời đúng (ground truth): {true_answer}")

    # st.markdown("---")
    # st.subheader("🧪 Tự thử với ảnh và câu hỏi của bạn")

    # uploaded_image = st.file_uploader("Tải ảnh lên", type=["png", "jpg", "jpeg"])
    # custom_question = st.text_input("Nhập câu hỏi")

    # if uploaded_image and custom_question:
    #     img = Image.open(uploaded_image).convert("RGB")
    #     st.image(img, caption="Ảnh của bạn", width=400)
    #     if st.button("Dự đoán"):
    #         pred = predict(img, custom_question, model, processor, device=device)
    #         st.success(f"📌 Câu trả lời: {pred}")
