import os
import sys
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import torch
from torchvision import models
from torchvision import transforms

from dotenv import load_dotenv
import torchvision

load_dotenv(dotenv_path="system/src/core/config/.env")

project_root = os.getenv("PROJECT_ROOT")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from system.src.interface.models.deeplabv3_plus import Deeplabv3Plus


def load_model(model_path):
    model = Deeplabv3Plus(num_classes=1)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0), image


import cv2
import numpy as np

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import mahotas
from io import BytesIO
import matplotlib.pyplot as plt

def analyze_lesion_details(binary_mask, original_image):
    """
    binary_mask: np.array, nhị phân (0/1) kích thước (H, W)
    original_image: PIL.Image RGB

    Trả về dict các phân tích chi tiết.
    """

    details = {}

    lesion_area = np.sum(binary_mask)
    total_area = binary_mask.shape[0] * binary_mask.shape[1]
    lesion_ratio = lesion_area / total_area * 100
    details['lesion_area'] = lesion_area
    details['lesion_ratio'] = lesion_ratio

    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
    details['perimeter'] = perimeter

    moments = cv2.moments(binary_mask.astype(np.uint8))
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = None, None
    details['centroid'] = (cx, cy)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        circ = (4 * np.pi * cv2.contourArea(cnt)) / (cv2.arcLength(cnt, True) ** 2) if perimeter > 0 else 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0

        rect_area = w * h
        extent = cv2.contourArea(cnt) / rect_area if rect_area > 0 else 0
    else:
        aspect_ratio = circ = solidity = extent = 0

    details['aspect_ratio'] = aspect_ratio
    details['circularity'] = circ
    details['solidity'] = solidity
    details['extent'] = extent

    img_np = np.array(original_image.resize((binary_mask.shape[1], binary_mask.shape[0])))
    lesion_pixels = img_np[binary_mask == 1]

    if len(lesion_pixels) > 0:
        mean_color = lesion_pixels.mean(axis=0)
        median_color = np.median(lesion_pixels, axis=0)
        std_color = lesion_pixels.std(axis=0)
    else:
        mean_color = median_color = std_color = np.array([0,0,0])

    details['mean_color'] = mean_color
    details['median_color'] = median_color
    details['std_color'] = std_color

    gray_img = rgb2gray(img_np)
    gray_img = (gray_img * 255).astype(np.uint8)
    gray_lesion = gray_img * binary_mask

    glcm = graycomatrix(gray_lesion, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0,0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    details['texture_contrast'] = contrast
    details['texture_homogeneity'] = homogeneity
    details['texture_entropy'] = entropy

    lbp = local_binary_pattern(gray_lesion, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    details['lbp_histogram'] = hist

    num_labels, labels_im = cv2.connectedComponents(binary_mask.astype(np.uint8))
    details['num_lesions'] = num_labels - 1

    flipped = np.fliplr(binary_mask)
    intersection = np.logical_and(binary_mask, flipped).sum()
    union = np.logical_or(binary_mask, flipped).sum()
    symmetry = intersection / union if union > 0 else 0
    details['symmetry'] = symmetry

    def fractal_dimension(Z):
        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                   np.arange(0, Z.shape[1], k), axis=1)
            return len(np.where((S > 0) & (S < k*k))[0])

        Z = (Z > 0)
        p = min(Z.shape)
        n = 2**np.floor(np.log2(p))
        n = int(n)
        sizes = 2**np.arange(int(np.log2(n)), 1, -1)
        counts = []
        for size in sizes:
            counts.append(boxcount(Z, size))
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0] if len(counts) > 1 else 0

    fractal_dim = fractal_dimension(binary_mask)
    details['fractal_dimension'] = fractal_dim

    ys, xs = np.where(binary_mask)
    if len(xs) > 0 and cx is not None and cy is not None:
        distances = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        mean_distance = distances.mean()
        std_distance = distances.std()
    else:
        mean_distance = std_distance = 0
    details['mean_distance_to_centroid'] = mean_distance
    details['std_distance_to_centroid'] = std_distance

    return details


def predict_and_plot(model, input_tensor, original_image):
    import torch

    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)

    resized_img = original_image.resize((mask.shape[1], mask.shape[0]))
    original_np = np.array(resized_img)

    details = analyze_lesion_details(binary_mask, resized_img)

    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    axs[0].imshow(original_np)
    axs[0].set_title("Hình ảnh gốc")
    axs[0].axis("off")

    axs[1].imshow(binary_mask, cmap='gray')
    axs[1].set_title("Khoanh vùng tổn thương")
    axs[1].axis("off")

    axs[2].imshow(original_np)
    axs[2].contour(binary_mask, colors='yellow', linewidths=2)
    cx, cy = details['centroid']
    if cx and cy:
        axs[2].plot(cx, cy, 'ro', label='Centroid')
        axs[2].legend()
    axs[2].set_title("Kết quả phân đoạn tổn thương (predicted mask)")
    axs[2].axis("off")

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return buf, details



import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

@st.cache_resource
def load_classify_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    num_classes = 7  
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

# Tiền xử lý ảnh
def preprocess_classify_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  

label_map = {
    0: "MEL: Melanoma",
    1: "NV: Melanocytic nevus",
    2: "BCC: Basal cell carcinoma",
    3: "AKIEC: Actinic keratosis",
    4: "BKL: Benign keratosis-like lesions",
    5: "DF: Dermatofibroma",
    6: "VASC: Vascular lesions"
}



import pandas as pd
import os

@st.cache_data
def load_ground_truth_labels(csv_path):
    df = pd.read_csv(csv_path)  

    df.set_index("image", inplace=True)


    label_dict = {}
    for image_id, row in df.iterrows():
        class_index = row.values.argmax()  
        label_dict[image_id] = class_index
    return label_dict

label_csv_path = "storage/ISIC2018_Task3_Training_GroundTruth.csv"
ground_truth_labels = load_ground_truth_labels(label_csv_path)
