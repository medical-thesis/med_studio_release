import os
import streamlit as st
import io
import random
from PIL import Image
import torch

from system.src.interface.controllers.chat_with_image import (
    load_model,
    predict_and_plot,
    preprocess_image,
    label_map,
    load_classify_model,
    preprocess_classify_image,
    ground_truth_labels
)


def generate_diagnosis(query: str) -> str:
    return f"Phân tích bệnh lý từ câu hỏi: \"{query}\""


def init_session_state():
    if 'selected_sample_image' not in st.session_state:
        st.session_state.selected_sample_image = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None


def on_image_click(img_path):
    st.session_state.selected_sample_image = img_path
    st.session_state.uploaded_file = None


def render():
    init_session_state()
    st.subheader("📬 DermatoAI Studio: Messenger with image")
    st.markdown(
        "Vui lòng nhập câu hỏi và tải ảnh hoặc chọn ảnh mẫu để hệ thống phân tích.")

    st.subheader("🎲 Ảnh mẫu ngẫu nhiên từ thư mục dữ liệu hệ thống")

    image_dir = r"storage/ISIC2018_Task3_Training_Input"
    all_images = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sample_images = random.sample(all_images, min(20, len(all_images)))

    cols = st.columns(5)
    for idx, img_path in enumerate(sample_images):
        with cols[idx % 5]:
            try:
                img = Image.open(img_path)
                st.image(img, use_container_width=True,
                         caption=os.path.basename(img_path))
                st.button(
                    "Thử với mẫu này",
                    key=f"btn_{os.path.basename(img_path)}",
                    on_click=on_image_click,
                    args=(img_path,)
                )
            except Exception as e:
                st.error(f"Lỗi khi tải ảnh: {e}")

    with st.form("diagnosis_form"):
        query = st.text_input("Câu hỏi về bệnh lý")

        if st.session_state.selected_sample_image:
            st.success(
                f"✅ Đã chọn ảnh mẫu: {os.path.basename(st.session_state.selected_sample_image)}")
            if st.form_submit_button("Hủy chọn ảnh mẫu"):
                st.session_state.selected_sample_image = None
                st.rerun()

        if not st.session_state.selected_sample_image:
            uploaded_file = st.file_uploader(
                "Tải ảnh vùng da tổn thương",
                type=["png", "jpg", "jpeg"],
                key="file_uploader"
            )
        else:
            uploaded_file = None

        submitted = st.form_submit_button("Phân tích")

    if submitted:
        if not query: pass

        if not st.session_state.selected_sample_image and not uploaded_file:
            st.warning("❗ Bạn cần tải ảnh hoặc chọn ảnh mẫu.")
            return

        if st.session_state.selected_sample_image:
            try:
                with open(st.session_state.selected_sample_image, "rb") as f:
                    img_bytes = f.read()
                image_io = io.BytesIO(img_bytes)
                image = Image.open(image_io).convert('RGB')
                image_name = os.path.splitext(os.path.basename(
                    st.session_state.selected_sample_image))[0]
                st.image(
                    image, caption=f'Ảnh mẫu: {image_name}', use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi khi đọc ảnh mẫu: {e}")
                return
        else:
            img_bytes = uploaded_file.read()
            image_io = io.BytesIO(img_bytes)
            image = Image.open(image_io).convert('RGB')
            image_name = os.path.splitext(uploaded_file.name)[0]
            st.image(image, caption='Hình ảnh đã tải lên ..',
                     use_container_width=True)

        with st.spinner("Đang phân loại tổn thương..."):
            try:
                model = load_classify_model(
                    r"storage/resnet18_isic2018_5_epochs.pth")
                input_tensor = preprocess_classify_image(image)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    class_id = predicted.item()
                    class_name = label_map.get(class_id, "Không rõ")
            except Exception as e:
                st.error(f"Lỗi khi phân loại: {e}")
                return

        true_class_id = ground_truth_labels.get(image_name, None)
        true_class_name = label_map.get(
            true_class_id, "Không rõ") if true_class_id is not None else "Không rõ"

        st.success(
            f"📌 Kết quả phân loại tổn thương: **{class_name}** (Lớp {class_id})")
        st.info(
            f"✅ Ground truth của tổn thương: **{true_class_name}** (Lớp {true_class_id})")

        st.info("🔍 Đang phân tích hình ảnh...")
        try:
            model = load_model(
                "storage/deeplabv3_plus_pretrained_model_isic_2018.pth.tar")
            input_tensor, original_image = preprocess_image(image_io)
            result_buf, details = predict_and_plot(
                model, input_tensor, original_image)
            segmented_image_bytes = result_buf.getvalue()
        except Exception as e:
            st.error(f"Lỗi khi phân đoạn ảnh: {e}")
            return

        st.subheader("📊 Thống kê tổn thương chi tiết")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Thông số cơ bản**")
            st.markdown(
                f"- Diện tích tổn thương: **{details['lesion_area']} pixels**")
            st.markdown(
                f"- Tỷ lệ tổn thương: **{details['lesion_ratio']:.2f}%**")
            st.markdown(
                f"- Chu vi vùng tổn thương: **{details['perimeter']:.2f} pixels**")
            st.markdown(
                f"- Số vùng tổn thương riêng biệt: **{details['num_lesions']}**")
            st.markdown(f"- Aspect ratio: **{details['aspect_ratio']:.2f}**")

        with col2:
            st.markdown("**Đặc điểm hình dạng**")
            st.markdown(f"- Circularity: **{details['circularity']:.4f}**")
            st.markdown(f"- Solidity: **{details['solidity']:.4f}**")
            st.markdown(f"- Extent: **{details['extent']:.4f}**")
            st.markdown(
                f"- Đối xứng vùng tổn thương: **{details['symmetry']:.4f}**")
            st.markdown(
                f"- Fractal dimension: **{details['fractal_dimension']:.4f}**")

        with col3:
            st.markdown("**Thông số khác**")
            if details['centroid'][0] is not None:
                st.markdown(
                    f"- Tâm vùng tổn thương: **({details['centroid'][0]}, {details['centroid'][1]})**")
            st.markdown(
                f"- Mean distance to centroid: **{details['mean_distance_to_centroid']:.2f}**")
            st.markdown(
                f"- Std distance to centroid: **{details['std_distance_to_centroid']:.2f}**")
            mean_color = details['mean_color']
            st.markdown(
                f"- Màu trung bình vùng tổn thương (RGB): **({mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f})**")
            st.markdown(f"- Contrast: **{details['texture_contrast']:.4f}**")
            st.markdown(
                f"- Homogeneity: **{details['texture_homogeneity']:.4f}**")
            st.markdown(f"- Entropy: **{details['texture_entropy']:.4f}**")

        diagnosis_text = generate_diagnosis(query)
        st.subheader("📝 Phân tích bệnh lý")
        st.markdown(diagnosis_text)

        st.subheader("🖼️ Kết quả phân đoạn tổn thương")
        st.image(segmented_image_bytes,
                 caption="Hình ảnh kết quả phân đoạn", use_container_width=True)
