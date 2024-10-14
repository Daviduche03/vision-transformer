import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from PIL import Image
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from datasets import load_dataset
import requests
from io import BytesIO
from tqdm import tqdm
import random
import modal
import os
from huggingface_hub import HfApi

# Initialize the Modal App
stub = modal.App("vit_model")

# Define the image stub
image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "datasets",
    "pillow",
    "huggingface_hub",
    "tqdm",
    "transformers",
    "einops"
)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = torch.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0., forward_expansion=4, forward_drop_p=0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, n_classes=2):
        super().__init__(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, n_classes=2, **kwargs):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        self.classification_head = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.classification_head(x[:, 0, :])  # Only use the CLS token for classification
        return x

def create_ai_image_detector(pretrained=False):
    model = ViT(n_classes=2)  # Ensure we're using 2 classes
    if pretrained:
        # Load pretrained weights if available
        # model.load_state_dict(torch.load('path_to_pretrained_weights.pth'))
        pass
    return model



class URLImageDataset(Dataset): 
    def __init__(self, hf_dataset, label, source, transform=None): 
        self.dataset = hf_dataset 
        self.label = label
        self.transform = transform
        self.source = source

    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.source == "ai":
            image_url = item['image']
        else:
            image_url = item['image_url']

        image = None
        try:
            response = requests.get(image_url) 
            image = Image.open(BytesIO(response.content)).convert('RGB') 
        except Exception as e: 
            print(f"Failed to fetch image {image_url}: {e}")
            
            # Retry with the modified URL if the original one failed
            if "sm2_webp" in image_url:
                image_url = image_url.replace("sm2_webp", "full_webp")
                try:
                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as retry_e:
                    print(f"Retry failed with {image_url}: {retry_e}")
                    image = Image.new('RGB', (224, 224), (255, 255, 255))  # Placeholder image
            else:
                image = Image.new('RGB', (224, 224), (255, 255, 255))  # Placeholder image

        if self.transform and image is not None: 
            image = self.transform(image)

        return image, self.label  # Return the image and integer label



def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")
        print("--------------------")
        return model
    
    
@stub.function(
    image=image,
    gpu="H100:2",
    timeout=86400,  # 24 hours
    cpu=8,
    secrets=[modal.Secret.from_name("my-huggingface-secret")],
)
def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ai_dataset = load_dataset('Daviduche03/AI-Generated-vs-Real-Images-Datasets', token=os.environ["HF_TOKEN"])
    real_dataset = load_dataset('SilentAntagonist/vintage-photography-450k-high-quality-captions')

    num_of_images = 163000

    indices = random.sample(range(len(real_dataset['train'])), num_of_images)
    sampled_images = real_dataset['train'].select(indices)
    real_dataset['train'] = sampled_images

    ai_train_dataset = URLImageDataset(ai_dataset['train'], label=0, transform=transform, source="ai")
    real_train_dataset = URLImageDataset(real_dataset['train'], label=1, transform=transform, source="real")

    # Create smaller validation sets
    ai_val_size = min(len(ai_dataset['train']) // 10, 10000)  # 10% or max 10000
    real_val_size = min(len(real_dataset['train']) // 10, 10000)  # 10% or max 10000

    ai_val_dataset = URLImageDataset(ai_dataset['train'].select(range(ai_val_size)), label=0, transform=transform, source="ai")
    real_val_dataset = URLImageDataset(real_dataset['train'].select(range(real_val_size)), label=1, transform=transform, source="real")

    train_dataset = ConcatDataset([ai_train_dataset, real_train_dataset])
    val_dataset = ConcatDataset([ai_val_dataset, real_val_dataset])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = create_ai_image_detector().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    torch.save(model.state_dict(), 'vit_ai_detector.pth')
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj="vit_ai_detector.pth",
            path_in_repo="vit_ai_detector.pth",
            repo_id=os.environ["HUGGINGFACE_REPO_ID"],
            token=os.environ["HF_TOKEN"]
        )

        print(f"Model uploaded to Hugging Face: https://huggingface.co/{os.environ['HUGGINGFACE_REPO_ID']}")
    except Exception as e:
        print(f"Error during model upload: {str(e)}")
    finally:
        if os.path.exists("vit_ai_detector.pth"):
            os.remove("vit_ai_detector.pth")
    
    print("Model saved as vit_ai_detector.pth")


@stub.local_entrypoint()
def entrypoint():
    print("Local entrypoint triggered. Starting classifier training and upload...")
    main.remote()
    print("Process initiated. Check Modal dashboard for progress.")

if __name__ == "__main__":
    modal.runner.deploy_app(stub)
    entrypoint()