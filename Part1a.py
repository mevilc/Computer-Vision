import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms 
import copy 

# transform data
data_transforms = transforms.Compose([transforms.Resize((224,224)),  
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
                                      ]) 

# import data and transform them
trainset = torchvision.datasets.CIFAR100(root="./data", train = True, download=True, transform=data_transforms) 
testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=data_transforms) 

# load data
train_data_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)

# model
model = models.vgg16(pretrained = True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 100)

for param in model.parameters():  
    param.requires_grad = False 
for param in model.classifier[6].parameters(): # train the last linear layer. 
    param.requires_grad = True 

# hyperparameters
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = model.to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# train model
for epoch in range(num_epochs):
    for i, data in enumerate(train_data_loader, 0):
        images, labels = data # data is a mini-batch input 
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(images) # here the model is the pretrained VGG16 
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 

    scheduler.step()

# save best model weights
best_model_wts = copy.deepcopy(model.state_dict()) 
torch.save(best_model_wts , 'best_model_weight.pth')

# Test model
model.load_state_dict(torch.load('best_model_weight.pth'))
model.eval()

for images, labels in test_data_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images) # here the model is the pretrained VGG16
    _, preds = torch.max(outputs, 1) # preds are our prediction
    accuracy = torch.tensor(torch.sum(preds == labels).item() / len(preds))

print(f"Test set accuracy: {accuracy}")