import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.optim import lr_scheduler 
import numpy as np 
import torchvision 
from torchvision import datasets, models, transforms 
import cv2

label_map = [
               (0, 0, 0),  # background
               (128, 0, 0), # aeroplane
               (0, 128, 0), # bicycle
               (128, 128, 0), # bird
               (0, 0, 128), # boat
               (128, 0, 128), # bottle
               (0, 128, 128), # bus 
               (128, 128, 128), # car
               (64, 0, 0), # cat
               (192, 0, 0), # chair
               (64, 128, 0), # cow
               (192, 128, 0), # dining table
               (64, 0, 128), # dog
               (192, 0, 128), # horse
               (64, 128, 128), # motorbike
               (192, 128, 128), # person
               (0, 64, 0), # potted plant
               (128, 64, 0), # sheep
               (0, 192, 0), # sofa
               (128, 192, 0), # train
               (0, 64, 128) # tv/monitor
]

data_transforms = transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.546, 0.406], [0.229, 0.224, 0.225]) 
                                      ]) 

def get_segment_labels(image, model, device):
    image = data_transforms(image).to(device)
    image = image.unsqueeze(0)
    outputs = model(image)
    return outputs

def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
        
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
            
    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image

model = models.segmentation.fcn_resnet50(pretrained = True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval().to(device)

for i in range(1, 6):
    image = cv2.imread('Original/Im' + str(i) + '.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    outputs = get_segment_labels(image, model, device)
    # get the data from the `out` key
    outputs = outputs['out']

    segmented_image = draw_segmentation_map(outputs)
    cv2.imwrite('Seg/Im' + str(i) + ' - Seg', segmented_image)