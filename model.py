import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from architecture import VGGFeatureExtractor, gram_matrix, MyModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
def load_image(img_path, max_size=512):
    image = Image.open(img_path)
    size = max_size if max(image.size) > max_size else max(image.size)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Save trained model and gram matrices
def save_model(style_name, model, gram_matrices):
    os.makedirs("trained_styles", exist_ok=True)  # Ensure directory exists
    model_path = f"trained_styles/{style_name}_style.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "gram_matrices": gram_matrices
    }
    torch.save(checkpoint, model_path)
    print(f"Model and Gram matrices saved as {model_path}")

# Style transfer training loop
def train_style_transfer(content_image_path, style_image_path, style_name, epochs=100, style_weight=1e8, content_weight=1e0):
    epochs = 100
    # Load content and style images
    content_image = load_image(content_image_path).to(device)
    style_image = load_image(style_image_path).to(device)

    # Initialize VGG for feature extraction and the custom model
    vgg = VGGFeatureExtractor().to(device).eval()
    model = MyModel().to(device)

    # Extract style features (from multiple layers)
    style_features = vgg(style_image)
    gram_style = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Content loss (from a deeper layer, e.g., 'relu4_1')
    content_features = vgg(content_image)['relu4_1']

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        generated_image = model(content_image)
        generated_features = vgg(generated_image)

        # Compute content loss
        content_loss = torch.mean((generated_features['relu4_1'] - content_features) ** 2)

        # Compute style loss using multiple Gram matrices
        style_loss_value = 0
        for layer in gram_style:
            generated_gram = gram_matrix(generated_features[layer])
            style_loss_value += torch.mean((generated_gram - gram_style[layer]) ** 2)

        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss_value
        
        # Backward pass with retain_graph=True to allow multiple backward calls
        total_loss.backward(retain_graph=True)
        optimizer.step()

        # Print loss info every 50 epochs
        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{epochs}], Content Loss: {content_loss.item()}, Style Loss: {style_loss_value.item()}")

    # Save the trained model and Gram matrices
    save_model(style_name, model, gram_style)

# Main function to train three styles
if __name__ == "__main__":
    # Content image to use (replace with your actual content image path)
    content_image_path = "testImage1-styleTransfer.jpg"

    # Style images and corresponding style names
    style_images = {
        "VanGogh": "images\Vangogh-StyleTransfer.jpg",  # Replace with actual Van Gogh style image path
        "Salvador": "images\Salvador-Dali-Persistence-of-Memory-StyleTransfer.webp",  # Replace with actual Salvador Dali style image path
        "Leonardo": "images\Monalisa-StyleTransfer.jpg"  # Replace with actual Leonardo da Vinci style image path
    }

    # Train each style and save the models
    for style_name, style_image_path in style_images.items():
        print(f"\nTraining style: {style_name}")
        train_style_transfer(content_image_path, style_image_path, style_name, epochs=100)
