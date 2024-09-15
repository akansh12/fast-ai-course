import torch
import time
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=25, save_name=None):
    model = model.to(device)
    best_iou = 0.0
    time_start = int(time.time())

    # Lists to store metrics for each epoch
    train_losses = []
    train_ious = []
    train_dices = []
    
    val_losses = []
    val_ious = []
    val_dices = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_samples = 0

        for (image_A, image_B), labels in train_loader:
            image_A, image_B, labels = image_A.to(device), image_B.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            inputs = torch.cat([image_A, image_B], dim=1)  # Concatenate along the channel dimension
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

            # Convert predictions to binary mask
            outputs = (outputs > 0.5).float()

            # Calculate IoU for each image in the batch
            intersection = (outputs * labels).sum(dim=[1, 2, 3])
            union = (outputs + labels).sum(dim=[1, 2, 3]) - intersection
            iou = intersection / (union + 1e-6)

            # Calculate Dice score
            dice = (2. * intersection) / (union + intersection + 1e-6)

            # Accumulate IoU, Dice, and sample count
            total_iou += iou.sum().item()
            total_samples += labels.size(0)
            total_dice += dice.sum().item()

        # Calculate epoch-level metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_iou = total_iou / total_samples
        epoch_dice = total_dice / total_samples

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}, DICE: {epoch_dice:.4f}')
        # Append training metrics for the epoch
        train_losses.append(epoch_loss)
        train_ious.append(epoch_iou)
        train_dices.append(epoch_dice)

        # Validate the model and get validation metrics
        val_loss, val_iou, val_dice = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)

        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, DICE: {val_dice:.4f}')
        print("-"*40)

        # Step the StepLR
        scheduler.step()

        if val_iou > best_iou:
            os.makedirs(f'../model_weights/{time_start}', exist_ok=True)
            best_iou = val_iou
            print('Saving the best model with IoU:', best_iou)
            torch.save(model.state_dict(), f'../model_weights/{time_start}/{save_name}_{time_start}_best.pth')
        
    print('Finished Training')

    #save metrics 
    metrics = {
        'train_losses': train_losses,
        'train_ious': train_ious,
        'train_dices': train_dices,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'val_dices': val_dices
    }
    torch.save(metrics, f'../model_weights/{time_start}/{save_name}_{time_start}_metrics.pth')

    # Return all metrics lists
    return train_losses, train_ious, train_dices, val_losses, val_ious, val_dices


def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for (image_A, image_B), labels in val_loader:
            image_A, image_B, labels = image_A.to(device), image_B.to(device), labels.to(device)
            inputs = torch.cat([image_A, image_B], dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Convert predictions to binary mask
            outputs = (outputs > 0.5).float()
            
            # Calculate IoU for each image in the batch
            intersection = (outputs * labels).sum(dim=[1, 2, 3])
            union = (outputs + labels).sum(dim=[1, 2, 3]) - intersection
            iou = intersection / (union + 1e-6)

            # Calculate Dice score
            dice = (2. * intersection) / (union + intersection + 1e-6)

            # Accumulate IoU, Dice, and sample count
            total_iou += iou.sum().item()
            total_samples += labels.size(0)
            total_dice += dice.sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    mean_iou = total_iou / total_samples
    mean_dice = total_dice / total_samples
    return epoch_loss, mean_iou, mean_dice
