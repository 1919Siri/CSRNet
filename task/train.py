import json
import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image  # Import for image loading

# Assuming your model is defined somewhere
from model import CSRNet  # Change this as per your model import

# Function to load the image
def load_image(img_path):
    # Load the image using PIL
    image = Image.open(img_path).convert('RGB')
    
    # Define your transformations (resize, normalization, etc.)
    transform = transforms.Compose([
        transforms.Resize((544, 932)),  # Resize to match your model input size
        transforms.ToTensor(),           # Convert to tensor
    ])
    
    # Apply transformations and return
    return transform(image)

# Function to get the corresponding .h5 file path
def get_h5_file_path(img_path):
    # Logic to get the corresponding .h5 file path from img_path
    base_name = os.path.basename(img_path).replace('.jpg', '.h5')  # Change '.jpg' if needed
    return os.path.join('D:\\Data Science\\CSRNet-pytorch\\ShanghaiTech_Crowd_Counting_Dataset\\part_A_final\\train_data\\h5_files', base_name)  # Update to your h5 files directory

# Function to load density data from .h5 file
def load_density_data(h5_file_path):
    with h5py.File(h5_file_path, 'r') as f:
        density = f['density'][:]  # Use the correct key in your .h5 files
    return density

# Train function
# Train function
def train(train_list, model, criterion, optimizer, epoch, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize the loss tracker
    for i, img_path in enumerate(train_list):
        # Load the image
        image = load_image(img_path).unsqueeze(0).to(device)  # Add batch dimension and move to device

        # Load corresponding density data
        h5_file_path = get_h5_file_path(img_path)  # Get the corresponding .h5 file path
        targets = load_density_data(h5_file_path)  # Load density targets
        
        # Convert targets to a tensor and move to device
        # Convert targets to a tensor and move to device
        target_tensor = torch.tensor(targets,
                                     dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions


        optimizer.zero_grad()  # Clear previous gradients
        output = model(image)  # Forward pass
        
        # Upsample the output to match the target tensor size
        output = nn.functional.interpolate(output, size=(566, 932), mode='bilinear', align_corners=False)

        loss = criterion(output, target_tensor)  # Compute loss
        
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
        
        running_loss += loss.item()  # Accumulate loss
        if i % 10 == 0:  # Change this to your preferred frequency
            print(f'Epoch [{epoch}], Step [{i}/{len(train_list)}], Loss: {loss.item():.4f}')

    # Return average loss for the epoch
    return running_loss / len(train_list)


# Main function
def main():
    # Load your training and validation file lists
    with open('part_A_train.json', 'r') as f:
        train_list = json.load(f)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, criterion, and optimizer
    model = CSRNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 100  # Adjust this as needed
    for epoch in range(num_epochs):
        avg_loss = train(train_list, model, criterion, optimizer, epoch, device)  # Get average loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')  # Print average loss
        if (epoch + 1) % 10 == 0:  # Save every 10 epochs
            torch.save(model.state_dict(), f'csrnet_epoch_{epoch + 1}.pth')

if __name__ == "__main__":
    main()
