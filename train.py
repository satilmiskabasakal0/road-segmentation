import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.unet import UNet
from dataset import RoadDataset

def dice_loss(pred, target, smooth=1e-4):
    pred = torch.sigmoid(pred)  # logits → probability
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))


def train_one_epoch(model,dataloader,optimizer,bce_fn,device):
    model.train()
    total_loss = 0

    loop = tqdm(dataloader)

    for images,masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        #preds = preds.squeeze(1) # [B,1,H,W] -> [B,H,W]

        loss = bce_fn(preds, masks) + dice_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():.4f}")

    return total_loss/len(dataloader)




def main():
    writer = SummaryWriter(log_dir="runs/unet_tensorboard")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256,512)),
        transforms.ToTensor(),
    ])

    dataset = RoadDataset(
        image_dir="data_road/training/image_2",
        mask_dir="data_road/training/gt_image_2",
        transform = transform
    )

    dataloader = DataLoader(dataset,batch_size=4,shuffle=True)
    model = UNet(in_channels=3,out_channels=1).to(device)
    bce_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=3e-4)

    for epoch in range(10):
        avg_loss = train_one_epoch(model, dataloader, optimizer, bce_fn, device)
        print(f"Epoch [{epoch+1}/10]  Avg Loss: {avg_loss:.4f}")

        writer.add_scalar("Loss/train", avg_loss, epoch)

        # Tahmin görsellerini logla
        model.eval()
        with torch.no_grad():
            images, masks = next(iter(dataloader))
            images = images.to(device)
            masks = masks.to(device)
            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()

            writer.add_images("Input", images, epoch)
            writer.add_images("Prediction", preds, epoch)
            writer.add_images("GroundTruth", masks, epoch)

    writer.close()

    torch.save(model.state_dict(),"unet_trained.pth")
    print("Model saved to unet_trained.pth")


if __name__ == "__main__":
    main()