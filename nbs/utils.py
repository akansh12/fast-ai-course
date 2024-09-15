import matplotlib.pyplot as plt
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_images(image_A, image_B, label, predictions=None):
    if predictions is not None:
        fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image_A.permute(1, 2, 0))
    ax[0].set_title("Image A (Before)")
    ax[1].imshow(image_B.permute(1, 2, 0))
    ax[1].set_title("Image B (After)")
    ax[2].imshow(label.squeeze(), cmap='gray')
    ax[2].set_title("Change Mask")
    if predictions is not None:
        ax[3].imshow(predictions.squeeze(), cmap='gray')
        ax[3].set_title("Predicted Mask")
    plt.show()

def compute_iou(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection) / (union + 1e-6)

def compute_dice(pred, target):
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

def visualize_test_results(model, test_loader, num_examples=5):
    model.eval()
    with torch.no_grad():
        for i, ((image_A, image_B), labels) in enumerate(test_loader):
            image_A, image_B, labels = image_A.to(device), image_B.to(device), labels.to(device)
            inputs = torch.cat([image_A, image_B], dim=1)
            outputs = model(inputs)
            outputs = (outputs > 0.5).float()

            # Plot the results
            for j in range(min(len(image_A), num_examples)):
                plt.figure(figsize=(15, 4))
                
                plt.subplot(1, 4, 1)
                plt.imshow(image_A[j].cpu().permute(1, 2, 0))
                plt.title("Image A (Before)")
                
                plt.subplot(1, 4, 2)
                plt.imshow(image_B[j].cpu().permute(1, 2, 0))
                plt.title("Image B (After)")

                plt.subplot(1, 4, 3)
                plt.imshow(labels[j].cpu().squeeze(), cmap='gray')
                plt.title("GT Change Mask")
                
                plt.subplot(1, 4, 4)
                plt.imshow(outputs[j].cpu().squeeze(), cmap='gray')
                iou = compute_iou(outputs[j], labels[j])
                dice = compute_dice(outputs[j], labels[j])
                plt.title("Pred(IoU: {:.4f}, DICE: {:.4f})".format(iou, dice))
                print(f"IoU: {iou:.4f}, DICE: {dice:.4f}")
                plt.show()
                if j == num_examples - 1:
                    return
