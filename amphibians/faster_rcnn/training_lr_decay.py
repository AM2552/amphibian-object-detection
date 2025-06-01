import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn_v2, 
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    ResNet50_Weights,
    MobileNet_V3_Large_Weights
)
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from dataset_class import AmphibianDataset, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Compose
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from torchmetrics.detection.mean_ap import MeanAveragePrecision

def collate_fn(batch):
    return tuple(zip(*batch))

def select_device():
    """
    If GPUs are available, lists their total and free memory,
    then returns the device (cuda:X) with the most free memory.
    Otherwise, returns 'cpu'.
    """
    if not torch.cuda.is_available():
        print("No GPU found. Using CPU.")
        return 'cpu'

    free_mem_list = []
    for i in range(torch.cuda.device_count()):
        free_mem, total_mem = torch.cuda.mem_get_info(i)  # returns free, total (in bytes)
        free_mem_list.append((i, free_mem, total_mem))

    # Pick the GPU with the largest free memory
    best_gpu = max(free_mem_list, key=lambda x: x[1])[0]

    print("Available GPUs:")
    for i, free_mem, total_mem in free_mem_list:
        print(f"  GPU {i}: free={free_mem/1e9:.2f}GB total={total_mem/1e9:.2f}GB")
    print(f"--> Selecting GPU {best_gpu} with {free_mem_list[best_gpu][1]/1e9:.2f}GB free\n")

    return f"cuda:{best_gpu}"

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        trainable_backbone_layers=5
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Training loop
def train(model, data_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0.0
    pbar = tqdm(data_loader, desc=f'Train {epoch+1}', leave=True)
    for images, targets in pbar:
        filtered = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
        if not filtered:
            continue
        
        images, targets = zip(*filtered)
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
        pbar.set_postfix(loss=train_loss / len(data_loader))
    return train_loss / len(data_loader)

# Evaluation loop
def evaluate(model, data_loader, device, epoch):
    model.eval()
    eval_loss = 0.0
    pbar = tqdm(data_loader, desc=f'Eval {epoch+1}', leave=True)
    with torch.no_grad():
        for images, targets in pbar:
            filtered = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
            if not filtered:
                continue
            
            images, targets = zip(*filtered)
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Temporarily set model to train to compute the loss in evaluation
            model.train()
            eval_loss_dict = model(images, targets)
            model.eval()

            eval_losses = sum(loss for loss in eval_loss_dict.values())
            eval_loss += eval_losses.item()
            pbar.set_postfix(loss=eval_loss / len(data_loader))
    return eval_loss / len(data_loader)

def evaluate_map(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    with torch.no_grad():
        for images, targets in data_loader:
            # Filter out any samples with no bounding boxes
            filtered = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
            if not filtered:
                continue
            
            images, targets = zip(*filtered)
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Inference
            preds = model(images)
            
            # Prepare for torchmetrics
            # Each element in preds/targets is a dict containing 'boxes', 'labels', 'scores' (preds only)
            # boxes, labels should all be on CPU for torchmetrics
            formatted_preds = []
            for pred in preds:
                formatted_preds.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
            
            formatted_targets = []
            for tgt in targets:
                formatted_targets.append({
                    'boxes': tgt['boxes'].cpu(),
                    'labels': tgt['labels'].cpu()
                })

            # Update metric
            metric.update(formatted_preds, formatted_targets)

    result = metric.compute()
    # result['map'] is overall mAP
    # result['map_50'] is mAP @ IoU=0.5
    return result['map'].item(), result['map_50'].item()

def plot_metrics(train_losses, val_losses, map_overall_list, map_50_list, save_dir):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(8, 8))

    # Upper plot for mAP
    plt.subplot(2, 1, 1)
    plt.plot(epochs, map_50_list, label='mAP@0.5')
    plt.plot(epochs, map_overall_list, label='mAP overall')
    plt.title('Mean Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()

    # Lower plot for Losses
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def main(pretrained_model_path):
    # 1. Select the best device (GPU with most free memory or CPU)
    device = select_device()

    # 2. Transforms and augmentations (example)
    transforms = T.Compose([
        T.ToTensor()
    ])
    augmentations = Compose([
        RandomHorizontalFlip(probability=0.5),
        RandomVerticalFlip(probability=0.5),
        RandomRotation(degrees=30)
    ])

    # 3. Datasets and Dataloaders (example usage)
    train_dataset = AmphibianDataset(
        annotations_file='amphibians/amphibia_annotations.json',
        img_dir='yolo/amphibia_molche/training',
        transforms=transforms,
        augmentations=augmentations
    )

    val_dataset = AmphibianDataset(
        annotations_file='amphibians/amphibia_annotations.json',
        img_dir='yolo/amphibia_molche/validation',
        transforms=transforms
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_fn
    )

    # 4. Get model, place on device
    model = get_model(num_classes=17)
    model.to(device)

    # 5. Define hyperparameters
    num_epochs = 80
    initial_lr = 0.01
    
    optimizer = SGD(model.parameters(), lr=initial_lr)
    exponentialLR_scheduler = ExponentialLR(optimizer, gamma=0.97)
    scheduler = exponentialLR_scheduler

    # 6. Load a pretrained checkpoint if requested
    if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    save_dir = 'amphibians/models/runXYZ'
    os.makedirs(save_dir, exist_ok=True)

    best_map = 0.0
    train_losses = []
    val_losses = []
    map_overall_list = []
    map_50_list = []

    # 7. Training loop
    for epoch in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{num_epochs} (Current LR: {scheduler.get_last_lr()[0]:.6f})")

        train_loss = train(model, train_loader, optimizer, device, epoch)
        train_losses.append(train_loss)

        validation_loss = evaluate(model, val_loader, device, epoch)
        val_losses.append(validation_loss)
        
        overall_map, map_05 = evaluate_map(model, val_loader, device)
        map_overall_list.append(overall_map)
        map_50_list.append(map_05)

        print(f"Training Loss: {round(train_loss, 5)} | "
              f"Validation Loss: {round(validation_loss, 5)} | "
              f"mAP@0.5: {round(map_05, 4)} | mAP Overall: {round(overall_map, 4)}")

        # 8. Adjust learning rate
        if epoch > 20:
            scheduler.step()

        # 9. Save the best model
        if overall_map > best_map:
            best_map = overall_map
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        # 10. Plot and save the learning curve at the end of **every epoch**
        # This way, if the training crashes, you still have the latest figure.
        plot_metrics(train_losses, val_losses, map_overall_list, map_50_list, save_dir)
    torch.save(model.state_dict(), os.path.join(save_dir, f"last_model.pt"))


if __name__ == '__main__':
    pretrained_model_path = None
    main(pretrained_model_path=pretrained_model_path)
