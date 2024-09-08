import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG19 model for feature extraction
vgg = models.vgg19(pretrained=True).features.to(device).eval()


# Helper Classes and Functions
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        self.relu = nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

def stylize_image(model, content_image_path, style_emphasis):
    content_image = load_image(content_image_path)
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    with torch.no_grad():
        output = model(content_image)
    
    # Resize output to match content image size if necessary
    if output.size() != content_image.size():
        output = nn.functional.interpolate(output, size=content_image.size()[2:], mode='bilinear', align_corners=False)
    
    output = output * style_emphasis + content_image * (1 - style_emphasis)
    
    # Convert the output tensor to a PIL Image
    output = output.squeeze(0).cpu().clamp(0, 1)
    output = transforms.ToPILImage()(output)
    
    return output

def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    return torch.mm(tensor, tensor.t()).div(b * c * h * w)

def train_style_model(style_image_path, num_epochs=500):
    num_epochs=300
    transformer = TransformerNet().to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()

    style_image = load_image(style_image_path)
    style_features = get_features(style_image, vgg, ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        output = transformer(style_image)
        output_features = get_features(output, vgg, ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'])
        
        style_loss = torch.tensor(0.0, device=device, requires_grad=True)
        for layer in style_features:
            output_gram = gram_matrix(output_features[layer])
            layer_style_loss = mse_loss(output_gram, style_grams[layer])
            style_loss = style_loss + layer_style_loss
        
        total_loss = style_loss
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}')

    return transformer

def stylize_image(model, content_image_path, style_emphasis):
    content_image = load_image(content_image_path)
    
    with torch.no_grad():
        output = model(content_image)
    
    # Resize output to match content image size if necessary
    if output.size() != content_image.size():
        output = nn.functional.interpolate(output, size=content_image.size()[2:], mode='bilinear', align_corners=False)
    
    output = output * style_emphasis + content_image * (1 - style_emphasis)
    
    output = output.squeeze(0).cpu().clamp(0, 1)
    return transforms.ToPILImage()(output)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path):
    model = TransformerNet()
    model.load_state_dict(torch.load(path))
    return model.to(device)

def main():
    # Hardcoded paths for style and input images
    path_to_style = 'images\Vangogh-StyleTransfer.jpg'
    path_to_input_image = 'vicky_test.jpeg'
    
    # Prompt the user for style emphasis
    style_emphasis = float(input('Enter style emphasis (0-1): '))
    
    print('Training style model...')
    style_model = train_style_model(path_to_style)
    
    print('Applying style transfer...')
    stylized_image = stylize_image(style_model, path_to_input_image, style_emphasis)
    
    output_path = 'stylized_output.jpg'
    stylized_image.save(output_path)
    print(f'Stylized image saved to {output_path}')

if __name__ == "__main__":
    main()