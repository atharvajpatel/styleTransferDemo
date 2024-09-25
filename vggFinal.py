import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
from torchvision.models import VGG19_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG19 for feature extraction
vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

# Define the layers to be used for content and style extraction
content_layers = [21]  # 'relu4_2' is layer 21
style_layers = [0, 5, 10, 19, 28]  # 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'

# Image loading and preprocessing
def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    
    if shape:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size),  # Resize to the correct size
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))  # Scale to [0, 255]
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Extract features from specific layers using indices
def get_features(image, model, layers):
    features = {}
    x = image
    for i, layer in enumerate(model.children()):
        x = layer(x)
        if i in layers:
            features[i] = x
    return features

# Compute the Gram matrix to represent style
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (d * h * w)

# Define a class to compute content and style losses
class StyleTransferModel:
    def __init__(self, content_img, style_img, content_weight, style_weight):
        print(content_weight)
        print(style_weight)
        self.content_img = content_img
        self.style_img = style_img
        self.content_weight = content_weight
        self.style_weight = style_weight

        # Extract content and style features
        self.content_features = get_features(self.content_img, vgg, content_layers)
        self.style_features = get_features(self.style_img, vgg, style_layers)
        self.style_grams = {layer: gram_matrix(self.style_features[layer]) for layer in self.style_features}

    def transfer(self, input_img, num_steps=500, learning_rate=1):
        # Optimizer
        optimizer = optim.Adam([input_img.requires_grad_()], lr=learning_rate)

        for step in range(num_steps):
            optimizer.zero_grad()

            # Get content and style features of the generated image
            generated_features = get_features(input_img, vgg, content_layers + style_layers)

            # Content loss
            content_loss = torch.mean((generated_features[21] - self.content_features[21]) ** 2)

            # Style loss
            style_loss = 0
            for layer in style_layers:
                generated_gram = gram_matrix(generated_features[layer])
                style_gram = self.style_grams[layer]
                layer_style_loss = torch.mean((generated_gram - style_gram) ** 2)
                style_loss += layer_style_loss

            # Total loss
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss

            # Backpropagation (retain the graph to avoid error)
            total_loss.backward(retain_graph=True)
            optimizer.step()

            # Display progress
            if step % 50 == 0:
                print(f"Step [{step}/{num_steps}], Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}")

        return input_img


# Convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor.cpu().clone().detach()
    tensor = tensor.squeeze(0)
    image = transforms.ToPILImage()(tensor / 255)
    return image

def main():
    # Use the paths you've provided
    content_image_path = 'images\horse.jpg' 
    style_image_path = 'images\Vangogh-StyleTransfer.jpg'

    # Load the content and style images at lower resolution
    content_img = load_image(content_image_path, max_size=256)
    style_img = load_image(style_image_path, max_size=256)

    # Initialize the input image to be the content image
    input_img = content_img.clone()

    # Ask the user to provide a value between 0 and 1 for style-content trade-off
    user_input = float(input("Enter a value between 0 and 1 (higher values give more style): "))
    
    # Cap the input to ensure it remains in the range [0, 1]
    user_input = max(0.0, min(1.0, user_input))

    # Define maximum content and style weights
    max_content_weight = 1e5
    max_style_weight = 1e10

    # Adjust content and style weights based on user input
    content_weight = (1 - user_input) * max_content_weight
    style_weight = user_input * max_style_weight

    # Initialize the style transfer model with adjusted weights
    model = StyleTransferModel(content_img, style_img, content_weight=content_weight, style_weight=style_weight)

    # Perform style transfer
    output_img = model.transfer(input_img)

    # Save the stylized image
    result = tensor_to_image(output_img)
    result.save('stylized_output_horse.jpg')
    print("Stylized image saved as 'stylized_output_horse.jpg'")


if __name__ == "__main__":
    main()
