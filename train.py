import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import random
from vit.vit import create_ai_image_detector
import modal
import os
from huggingface_hub import HfApi



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

