import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return preprocess(image)

data_select_path = '/home/st2/select/select_dataset_10'
batch_size = 100

for ipc_id in range(10):
    targets_all = torch.LongTensor(np.arange(1000))

    for kk in range(0, 1000, batch_size):

        start_index = kk
        end_index = min(kk + batch_size, 1000)

        inputs = []
        targets = targets_all[start_index:end_index].to('cuda')

        for folder_index in range(start_index, end_index):
            folder_path = os.path.join(data_select_path, os.listdir(data_select_path)[folder_index])
            image_path = os.path.join(folder_path, os.listdir(folder_path)[ipc_id])  
            
            image = load_image(image_path)  
            inputs.append(image)

        inputs_tensor = torch.stack(inputs).to('cuda')  
