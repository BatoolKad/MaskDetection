"""
Author: Batool Alkaddah 
PyCharm
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(32)

# Define data transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Specify the dataset path
dataset_path = "Dataset_project2"

# Load the dataset with transformations
dataset = datasets.ImageFolder(dataset_path)
dataset.transform = train_transform

# Define class labels
classes = ('Cloth mask', 'FFP2 Mask', 'No Mask', 'Surgery Mask')

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(16 * 16 * 64, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv_layer(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = self.fc_layer(x)
        return x

# Function to reset weights
def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

# Function to train the model
def train_model(model, device, train_loader, optimizer, epochs, criterion):
    model.train()
    loss_list = []
    accuracy_list = []

    for epoch in range(epochs):
        epoch_accuracy_list = []

        for i, (inputs, target) in enumerate(train_loader, 0):
            inputs = inputs.to(device)
            target = target.to(device)

            outputs = model(inputs)
            prediction = torch.max(outputs, 1)[1]
            epoch_accuracy_list.append(accuracy_score(target.cpu(), prediction.cpu()))

            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        accuracy = np.array(epoch_accuracy_list).mean()
        accuracy_list.append(accuracy)

        if (verbose):
            if ((epoch + 1) % print_every == 0):
                print('Epoch {}/{}: Loss {:.3f} Accuracy {:.3f}'.format(epoch + 1, epochs, loss.item(), accuracy))

    return loss_list, accuracy_list

# Function to test the model
def test_model(model, device, test_loader):
    model.eval()
    target_list = []
    prediction_list = []

    with torch.no_grad():
        for i, (inputs, target) in enumerate(test_loader, 0):
            inputs = inputs to(device)
            outputs = model(inputs).cpu()
            prediction = torch.max(outputs, 1)[1]
            target_list.append(target)
            prediction_list.append(prediction)

    target = np.hstack(target_list)
    prediction = np.hstack(prediction_list)
    report = classification_report(target, prediction, target_names=classes)
    print(report)
    precision_f = precision_score(target, prediction, average=None)
    accuracy_f = accuracy_score(target, prediction)
    recall_f = recall_score(target, prediction, average=None)
    f1_score_f = f1_score(target, prediction, average=None)
    return accuracy_f, precision_f, recall_f, f1_score_f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CNN().to(device)
net.apply(reset_weights)

batch_size = 32
folds = 10
epochs = 30
learning_rate = 0.001
kfold = KFold(n_splits=folds, shuffle=True)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss().to(device)
precision = []
recall = []
f1_score_l = []
accuracy = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
    print('------------Fold {}------------'.format(fold))
    train_subsampler = SubsetRandomSampler(train_idx)
    test_subsampler = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

    net.apply(reset_weights)
    loss, accuracy_ = train_model(net, device, train_loader, optimizer, epochs, criterion)
    accuracy_f, precision_f, recall_f, f1_score_f = test_model(net, device, test_loader)
    accuracy.append(accuracy_f)
    f1_score_l.append(f1_score_f)
    recall.append(recall_f)
    precision.append(precision_f)

print("Accuracy List:", accuracy)
print("Mean Accuracy:", np.mean(accuracy, axis=0))

print("F1 Score List:", f1_score_l)
print("Mean F1 Score:", np.mean(f1_score_l, axis=0))

print("Recall List:", recall)
print("Mean Recall:", np.mean(recall, axis=0))

print("Precision List:", precision)
print("Mean Precision:", np.mean(precision, axis=0))
