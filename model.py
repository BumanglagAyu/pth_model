import torch
import torch.nn as nn
from torchvision import models

# Class counts
SHAPE_ATTR_CLASSES = {
    "sleeve_length": 6,
    "lower_length": 5,
    "socks": 4,
    "hat": 3,
    "glasses": 5,
    "neckwear": 3,
    "wrist_wear": 3,
    "ring": 3,
    "waist_acc": 5,
    "neckline": 7,
    "outer": 3,
    "covers_navel": 3
}
FABRIC_CLASSES = 8
PATTERN_CLASSES = 8

# Model
class FashionRecognitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()

        self.shape_heads = nn.ModuleList([
            nn.Linear(2048, SHAPE_ATTR_CLASSES[key]) for key in SHAPE_ATTR_CLASSES
        ])
        self.fabric_heads = nn.ModuleList([
            nn.Linear(2048, FABRIC_CLASSES) for _ in range(3)  # upper, lower, outer
        ])
        self.pattern_heads = nn.ModuleList([
            nn.Linear(2048, PATTERN_CLASSES) for _ in range(3)
        ])

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flat(x)
        shape_out = [head(x) for head in self.shape_heads]
        fabric_out = [head(x) for head in self.fabric_heads]
        pattern_out = [head(x) for head in self.pattern_heads]
        return {'shape': shape_out, 'fabric': fabric_out, 'pattern': pattern_out}

# Prediction function
def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        shape_preds = [torch.argmax(out, dim=1).item() for out in outputs['shape']]
        fabric_preds = [torch.argmax(out, dim=1).item() for out in outputs['fabric']]
        pattern_preds = [torch.argmax(out, dim=1).item() for out in outputs['pattern']]

    attributes = {
        "fabric_upper": fabric_preds[0],
        "fabric_lower": fabric_preds[1],
        "fabric_outer": fabric_preds[2],
        "pattern_upper": pattern_preds[0],
        "pattern_lower": pattern_preds[1],
        "pattern_outer": pattern_preds[2],
        "sleeve_length": shape_preds[0],
        "lower_length": shape_preds[1],
        "socks": shape_preds[2],
        "hat": shape_preds[3],
        "glasses": shape_preds[4],
        "neckwear": shape_preds[5],
        "wrist_wear": shape_preds[6],
        "ring": shape_preds[7],
        "waist_acc": shape_preds[8],
        "neckline": shape_preds[9],
        "outer": shape_preds[10],
        "covers_navel": shape_preds[11]
    }

    return attributes
