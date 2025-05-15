from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from model import FashionRecognitionModel, predict
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import io

app = FastAPI()

# CORS middleware (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionRecognitionModel().to(device)
weights_path = "fashion_model.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print("✅ Model loaded.")
else:
    raise FileNotFoundError("❌ Model weights not found at 'fashion_model.pth'")

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/fashion-predict")
async def predict_fashion(
    bodyType: str = Form(...),
    gender: str = Form(...),
    prompt: str = Form(""),
    image: UploadFile = File(None)
):
    if image:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        attributes = predict(model, tensor)

        return {
            "bodyType": bodyType,
            "gender": gender,
            "prompt": prompt,
            "attributes": attributes
        }

    return {
        "bodyType": bodyType,
        "gender": gender,
        "prompt": prompt,
        "attributes": "No image provided — skipping CNN prediction"
    }
