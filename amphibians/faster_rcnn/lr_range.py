import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor, 
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    ResNet50_Weights,
    MobileNet_V3_Large_Weights
)
from torch.optim import Adam, SGD, AdamW
from dataset_class import AmphibianDataset, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        trainable_backbone_layers=5
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ---------------------- LR RANGE TEST FUNCTION ----------------------
def lr_range_test(
    model,
    data_loader,
    device,
    start_lr=1e-7,
    end_lr=1,
    num_iter=100,
    optimizer_cls=torch.optim.Adam,
    smoothing=0.05,
    diverge_threshold=5.0
):
    model.train()
    model.to(device)
    
    optimizer = optimizer_cls(model.parameters(), lr=start_lr)
    lr_factor = (end_lr / start_lr) ** (1 / num_iter)
    
    lrs = []
    losses = []
    
    best_loss = None
    avg_loss = 0.0
    current_lr = start_lr
    
    batch_iterator = iter(data_loader)
    
    for iteration in range(num_iter):
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            break
        
        filtered = [(img, tgt) for img, tgt in zip(images, targets) if len(tgt['boxes']) > 0]
        if not filtered:
            continue
        
        images, targets = zip(*filtered)
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if best_loss is None:
            best_loss = loss_val
            avg_loss = loss_val
        else:
            avg_loss = smoothing * loss_val + (1 - smoothing) * avg_loss
            if avg_loss < best_loss:
                best_loss = avg_loss

        if avg_loss > best_loss * diverge_threshold:
            print(f"Loss diverged at iteration {iteration+1}, stopping LR range test.")
            break
        
        lrs.append(current_lr)
        losses.append(avg_loss)

        current_lr *= lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

    return lrs, losses

def train(model, data_loader, optimizer, device, epoch):
    model.train()
    train_loss = 0.0
    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}', leave=True)
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

            # Trick to compute validation loss:
            model.train()
            eval_loss_dict = model(images, targets)
            model.eval()
            eval_losses = sum(loss for loss in eval_loss_dict.values())
            
            eval_loss += eval_losses.item()
            pbar.set_postfix(loss=eval_loss / len(data_loader))
    return eval_loss / len(data_loader)

def plot_losses(train_losses, val_losses, save_dir):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.show()

def main(pretrained_model_path, run_lr_test=False):
    
    transforms = T.Compose([
        T.ToTensor()
    ])

    augmentations = Compose([
        RandomHorizontalFlip(probability=0.5),
        RandomVerticalFlip(probability=0.5),
        RandomRotation(degrees=30)
    ])

    train_dataset = AmphibianDataset(
        annotations_file='amphibians/amphibia_annotations.json',
        img_dir='datasets/amphibia/training',
        transforms=transforms,
        augmentations=augmentations
    )

    val_dataset = AmphibianDataset(
        annotations_file='amphibians/amphibia_annotations.json',
        img_dir='datasets/amphibia/validation',
        transforms=transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = get_model(num_classes=17)
    model.to('cuda')

    if pretrained_model_path is not None and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path))

    if run_lr_test:
        # -- Run the LR range test before doing full training --
        print("Running LR range test...")
        
        # We can reinitialize a fresh model for the test if you want to keep your main model “untouched”.
        # Or you can run it on your existing model. Below, I reuse the same model:
        
        lrs, losses = lr_range_test(
            model,
            train_loader,
            device='cuda',
            start_lr=1e-7,
            end_lr=1,
            num_iter=200,
            optimizer_cls=SGD, 
            smoothing=0.05,
            diverge_threshold=5.0
        )
        
        # Plot the results (LR vs. Loss)
        plt.figure(figsize=(8,6))
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss (smoothed)")
        plt.title("LR Range Test")
        plt.savefig("lr_range_sgd_resnet.png")
        plt.show()
        
        # After seeing this plot, pick your best LR or range of LR
        return  # Exit after LR test. Remove 'return' if you want to continue training anyway.

    # Otherwise, do the usual training.
    optimizer = Adam(model.parameters(), lr=0.0001)

    num_epochs = 10
    save_dir = 'amphibians/models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    best_validation_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, 'cuda', epoch)
        train_losses.append(train_loss)

        validation_loss = evaluate(model, val_loader, 'cuda', epoch)
        val_losses.append(validation_loss)
        print(f"Training Loss: {round(train_loss, 5)} / Validation Loss: {round(validation_loss, 5)}")

        # save best model
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pth"))

    plot_losses(train_losses, val_losses, save_dir)

if __name__ == '__main__':
    pretrained_model_path = None
    
    # Set run_lr_test=True to do the LR finder test (and exit).
    # Set to False for normal training.
    main(pretrained_model_path=pretrained_model_path, run_lr_test=True)
