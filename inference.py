import torch
import cv2
import numpy as np
import argparse
from torchvision import transforms
import matplotlib.pyplot as plt
from model import myModel

IMAGE_PATH = "/home/trand/AI/TTAI_Segmentation/kaggle/input/LTE_frame_1000.png"

def load_model(model_path, num_classes, device):
    model = myModel(num_classes).to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def preprocess_image(image_path, transform):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB để hiển thị
    image_transformed = transform(image_rgb)
    image_transformed = image_transformed.unsqueeze(0)  # Thêm batch dimension
    return image_rgb, image_transformed


def inference(model, image, device):
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    return output


def display_images(original_image, predicted_mask):
    plt.figure(figsize=(10, 5))

    # Ảnh gốc bên trái
    plt.subplot(1, 2, 1)
    plt.title("Ảnh gốc")
    plt.imshow(original_image)
    plt.axis("off")

    # Ảnh dự đoán bên phải
    plt.subplot(1, 2, 2)
    plt.title("Kết quả dự đoán")
    plt.imshow(predicted_mask, cmap="jet")  # Dùng cmap="jet" để phân biệt các lớp
    plt.axis("off")

    plt.show()


def main():
    # Thiết lập argparse để nhận link ảnh từ command line
    parser = argparse.ArgumentParser(description="Inference script for semantic segmentation")
    parser.add_argument("--image_path", "-p", type=str, default=IMAGE_PATH)
    args = parser.parse_args()

    # Thiết lập device (GPU nếu có, nếu không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Một số tham số cần điều chỉnh theo mô hình của bạn
    num_classes = 3  # Số lớp trong dataset của bạn
    model_path = "checkpoints/22139078_22139044_best_model.pt"  # Đường dẫn đến file mô hình đã huấn luyện

    # Định nghĩa transform để tiền xử lý ảnh
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Kích thước đầu vào của mô hình
        transforms.ToTensor()
    ])

    # Load mô hình và thực hiện inference
    model = load_model(model_path, num_classes, device)
    original_image, image_transformed = preprocess_image(args.image_path, transform)
    output = inference(model, image_transformed, device)

    # Xử lý kết quả dự đoán
    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Hiển thị ảnh gốc và ảnh dự đoán
    display_images(original_image, pred)


if __name__ == "__main__":
    main()
