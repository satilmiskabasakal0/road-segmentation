from matplotlib import pyplot as plt
import torch
import matplotlib
matplotlib.use('TkAgg')


def show_batch(images,masks):
    """
    images : tensor of shape (B,3,H,W)
    masks : tensor of shape (B,1,H,W)
    """
    batch_size = images.size(0)

    for i in range(batch_size):
        img= images[i].permute(1,2,0).numpy() # CHW -> HWC
        mask = masks[i].squeeze().numpy()  # 1 x H x W -> H x W
        if mask.dtype != "float32":
            #print(mask.dtype)
            mask=mask.astype("float32")

        plt.figure(figsize=(10,10))

        # Original Image
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title("Inpute Image")
        plt.axis("off")

        # Ground Truth Mask
        plt.subplot(1,2,2)
        plt.imshow(mask,cmap='gray')
        plt.title(" Ground Truth Mask")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
