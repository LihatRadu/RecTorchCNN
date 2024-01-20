import customtkinter
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from customtkinter import CTkImage
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
from PIL import Image, ImageTk
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchviz

# Transform the data to tensors and normalize the pixel values to range [0, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Define the CNN class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust the input size for fc1
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Adjust the view size
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Create an instance of the CNN model
model = CNN()
torch.save(model.state_dict(), 'model_weights.pth')
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model():
    # Specify the number of training epochs
    epochs = 5

    # Train the model
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
        if epoch == 0 and i == 0:  # Visualize the graph only for the first batch of the first epoch
            # Get a single batch of data
            inputs, labels = next(iter(trainloader))
            # Forward pass to get the output
            outputs = model(inputs)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass to get the gradients
            loss.backward()
            # Visualize the graph
            torchviz.make_dot(loss, params=dict(model.named_parameters())).render("cnn_graph", format="png")


def test_model():
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_labels.extend(labels.numpy())
            predicted_labels.extend(predicted.numpy())

    accuracy = 100 * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")

    # Call the confusion matrix function
    graph_confusion_matrix(true_labels, predicted_labels)

def graph_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

def train_model_now():
    train_model()
    print("Model trained and updated successfully!")

def display_weights(image, weights, threshold=0.0):
    # Reshape the weights to match the shape of the image
    weights = weights.reshape(1, 32, 28, 28)

    # Normalize the weights
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    # Threshold the weights
    weights[weights < threshold] = 0

    # Add the weights to the image
    weighted_image = image + weights

    return weighted_image
def visualize_weights(model, image, layer_name):
    # Get the layer
    layer = model._modules.get(layer_name)

    # Get the weights
    weights = layer.weight.data

    # Apply the weights to the image
    feature_maps = layer(image)

    # Visualize the feature maps
    for i in range(feature_maps.size(0)):
        plt.imshow(feature_maps[i, 0, :, :], cmap='gray')
        plt.show()

def visualize_all_weights(model, image):
    # Visualize the weights of the first convolut
    # Visualize the weights of the first convolutional layer
    visualize_weights(model, image, 'conv1')

    # Visualize the weights of the second convolutional layer
    visualize_weights(model, image, 'conv2')
def display_images():
    # Create a new window
    image_window = customtkinter.CTkToplevel()
    image_window.geometry("1000x500")
    image_window.title("Training Images")
    customtkinter.set_appearance_mode("dark")

    # Create a frame
    image_frame = customtkinter.CTkFrame(image_window)
    image_frame.pack(pady=20, padx=20, fill="both", expand=True)

    # Display images
    for i, data in enumerate(trainloader, 0):
        if i >= 10:  # Display only the first 10 images
            break
        inputs, labels = data
        image = inputs[0]
        # Get the weights of the first convolutional layer
        weights = model.conv1.weight.detach().numpy()
        # Apply the weights to the image
        weighted_image = display_weights(image, weights)
        # Convert the image to a PIL Image
        pil_image = Image.fromarray((weighted_image * 255).astype(np.uint8))
        # Convert the PIL Image to a PhotoImage for Tkinter
        photo_image = ImageTk.PhotoImage(pil_image)
        # Create a label to display the image
        image_label = customtkinter.CTkLabel(image_frame, image=photo_image)
        image_label.image = photo_image  # Keep a reference to avoid garbage collection
        image_label.pack(side="left", padx=5)

    # Run the Tkinter event loop
    image_window.mainloop()

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Create a window
window = customtkinter.CTk()
window.geometry("1000x500")
window.title("Form Recognition in Images")
customtkinter.set_appearance_mode("dark")

frame = customtkinter.CTkFrame(window)
frame.pack(pady=20, padx=20, fill="both", expand=True)

title_panel = customtkinter.CTkLabel(frame, text="Form Recognition in Images", font=("Arial", 32), text_color="blue")
title_panel.pack(side="top")

button_results = customtkinter.CTkButton(frame, text="View Training Results", font=("Arial", 16), command=lambda: test_model())
button_results.pack(side="left", padx=20, pady=20)

button_train = customtkinter.CTkButton(frame, text="Train Now", font=("Arial", 16), command=train_model_now)
button_train.pack(side="left", padx=20, pady=20)

button_images = customtkinter.CTkButton(frame, text="View Training Images", font=("Arial", 16), command=display_images)
button_images.pack(side="left", padx=20, pady=20)

window.mainloop()