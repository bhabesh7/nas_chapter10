import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models

print("Defect Detection NAS Example")
# Build data paths relative to this script so the script can be run from any cwd
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

# Determine pretrained weights enums (torchvision >=0.13) and canonical mean/std.
try:
    # torchvision >= 0.13
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, MobileNet_V2_Weights

    res18_w = ResNet18_Weights.DEFAULT
    res34_w = ResNet34_Weights.DEFAULT
    mob_v2_w = MobileNet_V2_Weights.DEFAULT

    IMAGENET_MEAN = res18_w.meta["mean"]
    IMAGENET_STD = res18_w.meta["std"]
except Exception:
    # Older torchvision: fall back to classic imagenet stats
    res18_w = res34_w = mob_v2_w = None
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# data pipeline
# Use 3-channel mean/std for RGB images (match pretrained weights if available)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Validation transform: prefer the weights-provided transform when available
if res18_w is not None:
    # weights.transforms() contains the recommended resize/crop/normalize for eval
    val_transform = res18_w.transforms()
else:
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

print("Loading datasets...")

# Validate dataset directories before creating ImageFolder to give a clearer error
missing = []
if not TRAIN_DIR.exists():
    missing.append(str(TRAIN_DIR))
if not VAL_DIR.exists():
    missing.append(str(VAL_DIR))
if missing:
    raise FileNotFoundError(
        "Required data directories not found. Make sure the following paths exist:\n" + "\n".join(missing)
    )

train_dataset = datasets.ImageFolder(
    root=str(TRAIN_DIR),
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=str(VAL_DIR),
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# search space definition
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    
#defectnet
# class DefectNet(nn.Module):
#     def __init__(self, depth=3, width=32):
#         super().__init__()
#         layers = []
#         in_ch = 3
#         for _ in range(depth):
#             layers.append(ConvBlock(in_ch, width))
#             in_ch = width
#             width *= 2

#         self.features = nn.Sequential(*layers)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.classifier = nn.Linear(in_ch, 2)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x).view(x.size(0), -1)
#         return self.classifier(x)


# class DefectNet(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         backbone = models.resnet18(pretrained=True)
#         self.features = nn.Sequential(*list(backbone.children())[:-1])
#         self.classifier = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         return self.classifier(x)

from torchvision import models

class DefectNet(nn.Module):
    def __init__(self, backbone="resnet18", num_classes=2):
        super().__init__()

        if backbone == "resnet18":
            # prefer explicit weights enum when available, else fall back to old pretrained flag
            if res18_w is not None:
                print("Using ResNet18 pretrained weights enum.")
                model = models.resnet18(weights=res18_w)
            else:
                model = models.resnet18(pretrained=True)
            out_dim = 512
            self.features = nn.Sequential(*list(model.children())[:-1])

        elif backbone == "resnet34":
            if res34_w is not None:
                model = models.resnet34(weights=res34_w)
            else:
                model = models.resnet34(pretrained=True)
            out_dim = 512
            self.features = nn.Sequential(*list(model.children())[:-1])

        elif backbone == "mobilenet_v2":
            if mob_v2_w is not None:
                model = models.mobilenet_v2(weights=mob_v2_w)
            else:
                model = models.mobilenet_v2(pretrained=True)
            out_dim = 1280
            self.features = model.features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.classifier(x)



#hardware aware constraint function

def model_size_mb(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 ** 2)

MAX_MODEL_SIZE_MB = 100

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Batch Loss: {loss.item():.4f} label: {labels.tolist()}")
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

#nas loop
# search_space = [
#     {"depth": 3, "width": 32},
#     {"depth": 4, "width": 32},
#     {"depth": 3, "width": 64},
#     {"depth": 4, "width": 64},
# ]

search_space = [
    {"backbone": "resnet18"},
    {"backbone": "resnet34"},
    {"backbone": "mobilenet_v2"},
]


best_model = None
best_score = 0

for config in search_space:
    model = DefectNet(**config)
    print("\n")
    print(f"Evaluating config: {config}")
    print(45 * "-")
    print(f"Model size (MB): {model_size_mb(model):.2f}")

    if model_size_mb(model) > MAX_MODEL_SIZE_MB:
        continue  # hardware-aware pruning

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(5):  # short AutoML-style training
        train_one_epoch(model, train_loader, optimizer, criterion)

    acc = evaluate(model, val_loader)

    print(f"Config {config} → Accuracy: {acc:.3f}")

    if acc > best_score:
        best_score = acc
        best_model = model
        best_config = config
print(f"Accuracy of Best Model: {best_score:.3f}")
print(f"Size (MB) of Best Model: {model_size_mb(best_model):.2f}")
print(f"Config of Best Model: {best_config}")
# torch.save(best_model.state_dict(), "defect_detection_model.pth")
