import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms 
import copy 

data_transforms = transforms.Compose([transforms.Resize((224,224)),  
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
                                      ]) 

trainset = torchvision.datasets.CIFAR100(root="./data", train = True, download=True, transform=data_transforms) 
testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=data_transforms) 

train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            # input shape: # (224, 224, 3)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # (224, 224, 32)
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)) # (112, 112, 32)
        
        self.layer2 = nn.Sequential(
            # input shape: (112, 112, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # (112, 112, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # (56, 56, 64)
        
        self.layer3 = nn.Sequential(
            # input shape: (56, 56, 64)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (56, 56, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # (28, 28, 128)
        
        self.fc1 = nn.Linear(28 * 28 * 128, 1000)
        self.fc2 = nn.Linear(1000, 100)

    def forward(self, x):
        #print("1\n")
        out = self.layer1(x)
        #print("2\n")
        out = self.layer2(out)
        #print("3\n")
        out = self.layer3(out)
        #print("4\n")
        out = out.view(out.size(0), -1)
        #print("5\n")
        out = self.fc1(out)
        #print("6\n")
        out = self.fc2(out)
        return out

cnn = CNN()
cnn.train()
num_epochs = 30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
cnn = cnn.to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for epoch in range(num_epochs):
  for i, data in enumerate(train_data_loader, 0):
    images, labels = data # data is a mini-batch input 
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad() 
    outputs = cnn.forward(images)
    loss = criterion(outputs, labels) 
    loss.backward() 
    optimizer.step() 

  scheduler.step()

best_model_wts = copy.deepcopy(cnn.state_dict()) 
torch.save(best_model_wts , 'best_model_weight.pth')

cnn.load_state_dict(torch.load('best_model_weight.pth'))
cnn.eval()

for images, labels in test_data_loader:
  #images, labels = data # data is a mini-batch input
  images, labels = images.to(device), labels.to(device)
  outputs = cnn(images) # here the model is the pretrained VGG16
  _, preds = torch.max(outputs, 1) # preds are our prediction
  accuracy = torch.tensor(torch.sum(preds == labels).item() / len(preds))

print(f"Test set accuracy: {accuracy}")