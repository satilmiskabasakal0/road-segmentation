import cv2
import torch
import numpy as np
from torchvision import transforms
from models.unet import UNet
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet(3, 1).to(device)
model.load_state_dict(torch.load("unet_trained.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture("sample-dash-cam.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
combined_writer = cv2.VideoWriter('side_by_side_output_labeled.mp4', fourcc, fps, (width * 3, height))

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (255, 255, 255)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)
        pred_mask = (pred > 0.5).float().cpu().squeeze().numpy()

    mask_resized = cv2.resize(pred_mask, (width, height))
    mask_binary = (mask_resized * 255).astype(np.uint8)
    mask_colored = cv2.applyColorMap(mask_binary, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)


    def add_label(image, label):
        return cv2.putText(image.copy(), label, (10, 40), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    frame_labeled = add_label(frame, "Original")
    mask_labeled = add_label(mask_colored, "Mask")
    overlay_labeled = add_label(overlay, "Overlay")

    combined = np.hstack((frame_labeled, mask_labeled, overlay_labeled))
    combined_writer.write(combined)

cap.release()
combined_writer.release()
cv2.destroyAllWindows()

print("✅ Side-by-side video with labels saved as 'side_by_side_output_labeled.mp4'")
