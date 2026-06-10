import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = (512, 288)  # width, height


# -----------------------------
# U-Net Model Architecture
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.middle = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        mid = self.middle(p3)

        u3 = self.up3(mid)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return self.out(u1)


# -----------------------------
# Load trained model
# -----------------------------
def load_lane_model(model_path):
    model = UNetSmall()

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# -----------------------------
# Frame preprocessing
# -----------------------------
def preprocess_frame(frame):
    resized = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = torch.from_numpy(tensor).unsqueeze(0)

    return tensor.to(DEVICE)


# -----------------------------
# Predict lane mask
# -----------------------------
def predict_lane_mask(model, frame, threshold=0.5):
    original_h, original_w = frame.shape[:2]

    input_tensor = preprocess_frame(frame)

    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits)
        mask = prob.squeeze().cpu().numpy()

    mask = (mask > threshold).astype(np.uint8) * 255

    mask = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    return mask


# -----------------------------
# Overlay mask on frame
# -----------------------------
def overlay_lane(frame, mask):
    lane_color = np.zeros_like(frame)
    lane_color[:, :, 1] = mask

    overlay = cv2.addWeighted(frame, 0.8, lane_color, 0.6, 0)
    return overlay


# -----------------------------
# Process full video
# -----------------------------
def process_video(input_video, output_video, model_path):
    model = load_lane_model(model_path)

    cap = cv2.VideoCapture(str(input_video))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 25

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        mask = predict_lane_mask(model, frame)
        output_frame = overlay_lane(frame, mask)

        cv2.putText(
            output_frame,
            "Lane Detection",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        writer.write(output_frame)

        frame_count += 1

    cap.release()
    writer.release()

    return str(output_video), frame_count