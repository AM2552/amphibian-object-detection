import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import os

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
    #model = fasterrcnn_resnet50_fpn_v2(weights=None)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def load_model(model_path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, image_path, device='cuda'):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    model.to(device)
    with torch.no_grad():
        predictions = model(image_tensor)
    
    return predictions

def visualize_predictions(image_path, predictions, save_path, threshold=THRESHOLD):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    

    font_size = 15
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score >= threshold:
            draw.rectangle(
                [box[0].item(), box[1].item(), box[2].item(), box[3].item()],
                outline="red",
                width=2
            )
            
            text = f"{label_mapping[label.item()]}: {score.item():.2f}"
            text_width, text_height = draw.textsize(text, font=font)
            text_position = (box[0].item(), box[1].item())
            
            draw.rectangle(
                [text_position, (text_position[0] + text_width, text_position[1] + text_height)],
                fill="red"
            )
            draw.text(text_position, text, fill="white", font=font)
    
    image.save(save_path)


def process_directory(test_dir, model_path, output_dir):
    num_classes = 17
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(model_path, num_classes)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, _, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(subdir, file)
                save_path = os.path.join(output_dir, os.path.relpath(image_path, start=test_dir))
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                print(f"Processing {image_path}")
                
                predictions = predict(model, image_path, device)
                predicted_boxes = []
                for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
                    if score >= THRESHOLD:
                        predicted_boxes.append({
                            "xmin": box[0].item(),
                            "ymin": box[1].item(),
                            "xmax": box[2].item(),
                            "ymax": box[3].item(),
                            "label": label_mapping[label.item()],
                            "score": score.item()
                        })

                #print(json.dumps(predicted_boxes, indent=4, ensure_ascii=False))
                visualize_predictions(image_path, predictions, save_path)

if __name__ == "__main__":
    test_dir = 'datasets/amphibia/test'
    model_path = 'amphibians/models/best_mobilenet_320_pt_aug.pth'
    output_dir = 'amphibians/evaluation'
    process_directory(test_dir, model_path, output_dir)
