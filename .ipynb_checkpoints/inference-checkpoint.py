import matplotlib.pyplot as plt
import random
import torch
import numpy as np

# Hàm chuyển mask (nhãn) thành ảnh RGB
def mask_to_rgb(mask, class_colors):
    """Chuyển nhãn (label mask) thành ảnh RGB dựa trên class_colors."""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for rgb, idx in class_colors.items():
        rgb_mask[mask == idx] = rgb  # Gán màu theo class

    return rgb_mask


# Hàm lấy ngẫu nhiên một số mẫu từ bộ dữ liệu
def get_random_samples(dataloader, num_samples):
    """Lấy ngẫu nhiên một số ảnh từ bộ dữ liệu validation."""
    indices = random.sample(range(len(dataloader.dataset)), num_samples)  # Chọn ngẫu nhiên các chỉ số ảnh
    images_list, labels_list = [], []

    for idx in indices:
        image, label = dataloader.dataset[idx]  # Lấy ảnh và nhãn từ dataset theo chỉ số ngẫu nhiên
        images_list.append(image)
        labels_list.append(label)

    return torch.stack(images_list), torch.stack(labels_list)


# Hàm hiển thị ảnh, nhãn thực tế và nhãn dự đoán
def visualize_predictions(images, labels, preds, class_colors, num_samples=3):
    """Hiển thị ảnh đầu vào, nhãn thực tế và nhãn dự đoán."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))

    for j in range(num_samples):
        img = images[j].cpu().numpy().transpose(1, 2, 0)  # Chuyển Tensor về numpy
        lbl = labels[j].cpu().numpy()
        pred = preds[j].cpu().numpy()

        # Chuyển mask sang ảnh màu
        lbl_rgb = mask_to_rgb(lbl, class_colors)
        pred_rgb = mask_to_rgb(pred, class_colors)

        axes[j, 0].imshow(img)
        axes[j, 0].set_title("Input Image")
        axes[j, 0].axis("off")

        axes[j, 1].imshow(lbl_rgb)
        axes[j, 1].set_title("Ground Truth")
        axes[j, 1].axis("off")

        axes[j, 2].imshow(pred_rgb)
        axes[j, 2].set_title("Predicted Mask")
        axes[j, 2].axis("off")

    plt.show()


# Kiểm tra và hiển thị dự đoán
model = torch.jit.load("checkpoints/22139078_22139044_best_model.pt")
# epoch_loss_val, mAcc_val, mIoU_val = evaluate(model, val_dataloader, criterion, device, classes)
# print(f"Validation Loss: {epoch_loss_val:.4f}, Mean Accuracy: {mAcc_val:.4f}, Mean IoU: {mIoU_val:.4f}")
