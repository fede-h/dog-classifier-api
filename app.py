import io
import json
from pathlib import Path

import timm
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from PIL import Image, ImageOps
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

HOST = "127.0.0.1"
PORT = 8000

with open(MODEL_DIR / "class_names.json", "r") as f:
    class_names = json.load(f)

device = torch.device('cpu')
checkpoint = torch.load(MODEL_DIR / "efficientnet_b0.pt", map_location=device, weights_only=False)
model = timm.create_model(checkpoint.get("model_name", "efficientnet_b0"), pretrained=False, num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = FastAPI()

@app.get("/")
def home():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    filename = file.filename
    file_extension = filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")
    if not file_extension:
        raise HTTPException(status=415, detail="Unsupported file type")
    
    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    
    try:
        pil_image = Image.open(image_stream)
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
    except Exception as e:
        raise HTTPException(status=400, detail="Invalid image file")

    top_5 = predict_image(pil_image, topk=5)
    
    output = {}
    for class_name, prob in top_5:
        output[class_name] = prob

    return output

def predict_image(image: Image.Image, topk: int = 5):
    x = eval_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    
    values, indices = torch.topk(probs, k=min(topk, len(class_names)))
    return [(class_names[int(i)], float(v)) for v, i in zip(values, indices)]

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)