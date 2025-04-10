import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from dataset import RoadDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('TkAgg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")
# Loading Model
model = UNet(3, 1).to(device)

model.load_state_dict(torch.load("unet_trained.pth"))
model.eval()

# Dataset

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),

])

dataset = RoadDataset(
    "data_road/training/image_2",
    "data_road/training/gt_image_2",
    transform=transform,
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Prediction
iou_sum = 0.0
dice_sum = 0.0
num_batches = 0
def dice_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred= pred.squeeze().cpu().numpy()
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.item()

with torch.no_grad():
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        preds_s = torch.sigmoid(preds)  # Sigmoid output
        preds_st = (preds_s > 0.5).float()  # Binary Thresh Holding


        img = images[0].permute(1, 2, 0).cpu().numpy()
        true_mask = masks[0].squeeze().cpu().numpy()
        pred_mask = preds[0].squeeze().cpu().numpy()
        pred_s_mask = preds_s[0].squeeze().cpu().numpy()
        preds_st_mask = preds_st[0].squeeze().cpu().numpy()

        iou = iou_score(preds, masks)
        dice = dice_score(preds, true_mask)
        iou_sum += iou
        dice_sum += dice
        num_batches += 1
        print(f"""
---------------------
Batch {num_batches}
IoU: {iou:.4f}
Dice: {dice:.4f}
---------------------""")
        plt.figure(figsize=(15, 4))
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title("Input Image")

        plt.subplot(2, 3, 2)
        plt.imshow(true_mask)
        plt.title("Ground Truth")

        plt.subplot(2, 3, 3)
        plt.imshow(pred_mask)
        plt.title("Predicted Mask")

        plt.subplot(2, 3, 4)
        plt.imshow(pred_s_mask)
        plt.title("Sigmoid Prediction")

        plt.subplot(2, 3, 5)
        plt.imshow(preds_st_mask)
        plt.title("Sigmoid  Thresh Hold Prediction")

        plt.tight_layout()
        plt.show()

        #break  # For testing only 1 image not all batch


avg_iou = iou_sum / num_batches
avg_dice = dice_sum / num_batches

print(f"Average IoU: {avg_iou:.4f}, Average Dice Score: {avg_dice:.4f}")

