import torch
import torchvision.transforms as transforms
from PIL import Image


def predict_image(model, image_path):
    # Prepare the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Ensure model is in evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get the predicted class (0 for AI-generated, 1 for Real)
    predicted_class = torch.argmax(probabilities).item()

    # Get the confidence of the prediction
    confidence = probabilities[0, predicted_class].item()

    return "AI-generated" if predicted_class == 0 else "Real", confidence