import torch
from torchvision import transforms
from PIL import Image
from architecture import MyModel, VGGFeatureExtractor

# Preprocessing function for content and style images
def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')

    # Resize the image if it's larger than the max_size
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape:
        size = shape

    # Define the preprocessing steps: resize, convert to tensor, and scale
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))  # Scale to [0, 255]
    ])

    # Apply the transform and add a batch dimension
    image = transform(image)[:3, :, :].unsqueeze(0)  # Add batch dimension (1, C, H, W)
    return image

# Function to load the pre-trained models from the provided file path
def load_pretrained_model(style_name):
    """
    Load the pre-trained VGG model for feature extraction and the custom model for style transfer.
    
    Args:
    - style_name: The style to be applied (VanGogh, Salvador, Leonardo).
    
    Returns:
    - The loaded custom PyTorch model and VGG feature extractor with weights loaded.
    """
    # Mapping style names to the file paths of pre-trained models
    style_paths = {
        'VanGogh': 'trained_styles\\VanGogh_style.pth',
        'Salvador': 'trained_styles\\Salvador_style.pth',
        'Leonardo': 'trained_styles\\Leonardo_style.pth'
    }
    
    # Check if the selected style is valid
    if style_name not in style_paths:
        raise ValueError(f"Style '{style_name}' not recognized. Choose from VanGogh, Salvador, or Leonardo.")
    
    # Load the checkpoint for the style transfer model (which excludes VGG weights)
    checkpoint = torch.load(style_paths[style_name], map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Instantiate the custom model architecture
    model = MyModel()

    # Load the saved weights into the custom model (exclude VGG weights)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)  # Avoid loading the VGG weights
    
    # Set the custom model to evaluation mode
    model.eval()

    return model

# Function to apply style transfer using a pre-trained model
def apply_style(content_image_path, style_name, style_intensity):
    # Load the content image
    content_image = load_image(content_image_path)

    # Load the pre-trained style model
    model = load_pretrained_model(style_name)

    # Move content image to the appropriate device (if needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = content_image.to(device)
    model = model.to(device)

    # Forward pass to get the styled feature from the model
    with torch.no_grad():  # Disable gradients for inference
        styled_feature = model(content_image)

    # Blend the styled feature with the content image based on style intensity
    styled_feature = styled_feature * style_intensity ** 0.5 + content_image * (1 - style_intensity)

    # Post-processing to convert the tensor back into an image
    postprocess = transforms.Compose([
        transforms.Lambda(lambda x: x.mul(1 / 255)),  # Rescale back to [0, 1]
        transforms.ToPILImage()
    ])

    # Convert back to PIL image
    output_image = postprocess(styled_feature.squeeze(0).cpu())

    return output_image

# Main entry point of the program
if __name__ == '__main__':
    # Get user input for style intensity
    style_intensity = float(input("Enter style intensity (0.0 - 1.0): "))

    # Specify the content image and style name
    content_image_path = "testImage1-styleTransfer.jpg"  # Replace with the path to your image
    style_name = input("Enter style name (VanGogh, Salvador, Leonardo): ")

    # Apply the style transfer and save the output image
    try:
        output_img = apply_style(content_image_path, style_name, style_intensity)
        # Save or display the output image
        output_img.save("styled_output_image.jpg")
        output_img.show()
    except ValueError as e:
        print(e)
