from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from plots import plotTransformedImages

class CustomDatasetFromNumpyArray(Dataset):
    def __init__(self, data, target, transform=None, teste=None):
        
        #print('data',data)
        self.data = torch.from_numpy(data).float()
        #print('self.data', self.data)
        self.target = torch.from_numpy(target)
        self.transform = transform
        #self.customTransform = customTransform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        # if self.customTransform and (y == 1):
        #     #print('self.customTransform', self.customTransform, ' y = ', y)
        #     x = self.customTransform(x)
        #     #plotTransformedImages(x, index, 'teste_')
        # el
        if self.transform:
            #print('self.transform', self.transform)
            #print('x', x)
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)

class CustomTesteDatasetFromNumpyArray(Dataset):
    def __init__(self, data, target, transform=None):

        print('data',data)
        self.data = data#torch.from_numpy(data).float()
        print('self.data', self.data)
        self.target = torch.from_numpy(target)
        self.transform = transform
        #self.customTransform = customTransform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        # if self.customTransform and (y == 1):
        #     #print('self.customTransform', self.customTransform, ' y = ', y)
        #     x = self.customTransform(x)
        #     #plotTransformedImages(x, index, 'teste_')
        # el
        if self.transform:
            print('self.transform', self.transform)
            #print('x', x)
            print('x', x)
            input('aqui 1')
            x = self.transform(x)
            print('x', x)
            input('aqui 2')
            
        return x, y
    
    def __len__(self):
        return len(self.data)