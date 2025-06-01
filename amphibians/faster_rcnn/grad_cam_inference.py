import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

label_mapping = {
    1: 'feuersalamander',
    2: 'alpensalamander',
    3: 'bergmolch',
    4: 'kammmolch',
    5: 'teichmolch',
    6: 'rotbauchunke',
    7: 'gelbbauchunke',
    8: 'knoblauchkröte',
    9: 'erdkröte',
    10: 'kreuzkröte',
    11: 'wechselkröte',
    12: 'laubfrosch',
    13: 'moorfrosch',
    14: 'springfrosch',
    15: 'grasfrosch',
    16: 'wasserfrosch'
}

THRESHOLD = 0.5

def get_model(num_classes):
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_with_cam(model, image_path, device='cuda'):
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    np_img = np.array(image) / 255.0

    #target_layers = [model.backbone.body['13'].block[0][0]]
    target_layers = [model.backbone.body['16'][0]]

    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    with torch.no_grad():
        outputs = model(image_tensor)

    cams = []
    for idx, (box, label, score) in enumerate(zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores'])):
        if score >= THRESHOLD:
            targets = [FasterRCNNBoxScoreTarget(labels=label.unsqueeze(0), bounding_boxes=box.unsqueeze(0))]
            
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)

            cams.append((cam_image, box, label, score))
    
    return outputs, cams

def visualize_predictions_with_cam(image_path, outputs, cams, save_path, threshold=THRESHOLD):
    image = Image.open(image_path).convert("RGB")
    image = image.convert("RGBA")  # To support transparency
    overlay = Image.new('RGBA', image.size, (0,0,0,0))  # Empty overlay

    draw = ImageDraw.Draw(overlay)

    font_size = 15
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for cam_image, box, label, score in cams:
        cam_overlay = Image.fromarray(cam_image).convert("RGBA")
        image = Image.blend(image, cam_overlay, alpha=0.5)

        draw.rectangle(
            [box[0].item(), box[1].item(), box[2].item(), box[3].item()],
            outline="red",
            width=2
        )

        text = f"{label_mapping[label.item()]}: {score.item():.2f}"
        text_width, text_height = draw.textsize(text, font=font)
        text_position = (box[0].item(), box[1].item() - text_height)

        draw.rectangle(
            [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
            fill="red"
        )
        draw.text(text_position, text, fill="white", font=font)

    # Composite the overlay with the original image
    image = Image.alpha_composite(image.convert('RGBA'), overlay)
    image = image.convert('RGB')
    image.save(save_path)


def process_directory_with_cam(test_dir, model_path, output_dir):
    num_classes = 17
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(model_path, num_classes)
    print(model.backbone.body)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(subdir, file)
                save_path = os.path.join(output_dir, os.path.relpath(image_path, start=test_dir))

                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                print(f"Processing {image_path}")

                outputs, cams = predict_with_cam(model, image_path, device)
                visualize_predictions_with_cam(image_path, outputs, cams, save_path)

if __name__ == "__main__":
    test_dir = 'datasets/amphibia/test'
    model_path = 'amphibians/models/best_mobilenet_320_pt_aug.pth'
    output_dir = 'amphibians/evaluation'
    process_directory_with_cam(test_dir, model_path, output_dir)
