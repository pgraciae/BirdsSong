import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import from_numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import numpy as np
#librosa offset en el load

class BirdDataset(Dataset):
    '''
    spectograms dataset
    '''
    
    def __init__(self, data_path = '../data/', train = True, transform = None, target_transform = None):
        self.data_path = data_path
        self.metadata_path = os.path.join(data_path,'metadata.csv')
        # metadata = pd.read_csv(self.metadata_path, sep = ';') Falla la lectura de fitxers
        metadata = {'species': [],'spectrogram_name': []}

        #hardcode feo pero bueno
        for label_name in os.listdir(os.path.join(data_path,'spectrograms')):
            if os.path.isdir(os.path.join(data_path, 'spectrograms', label_name)):
                for file_name in os.listdir(os.path.join(data_path, 'spectrograms', label_name)):
                    metadata['species'].append(label_name)
                    metadata['spectrogram_name'].append(file_name)

        metadata = pd.DataFrame(metadata)
        self._coder = preprocessing.LabelEncoder()
        metadata['species_encoded'] = self._coder.fit_transform(metadata['species'])
        train_metadata, valid_metadata = train_test_split(metadata, test_size=0.2,
                                                                random_state=0,
                                                                stratify=metadata['species'])
        
        if train:
            self.metadata = train_metadata
        else:
            self.metadata = valid_metadata

        self.images_name = self.metadata['spectrogram_name'].to_numpy()
        self.labels_name = self.metadata['species'].to_numpy()
        self.labels = self.metadata['species_encoded'].to_numpy() # label encoder
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        label_name = self.labels_name[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.data_path, 'spectrograms', label_name, self.images_name[idx])
        
        image = np.load(img_path)
        if self.transform:
            image = self.transform(from_numpy(image).unsqueeze(0)).float()

        if self.target_transform:
            label = self.target_transform(label) #.long?

        return image, label

    @property
    def coder(self):
        return self._coder
