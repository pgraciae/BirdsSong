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
import librosa

#librosa offset en el load

class BirdDataset(Dataset):
    '''
    spectograms dataset
    '''
    def __init__(self, data_path = '../data/', transforms = {'data_transforms':None, 'target_transforms': None}):
        self.data_path = data_path
        self.metadata = {'species' : [], 'audio_name': [], 'audio_offset' : [], 'spectrograms_name' : []}
        for species in os.listdir(os.path.join(self.data_path, 'audio_files')):
            #si es fichero no hacer nada
            for audio in os.listdir(os.path.join(self.data_path, 'audio_files', species)):
                c = 0
                if 'txt' in audio:
                    with open(os.path.join(self.data_path, 'audio_files', species,audio)) as file:
                        for line in file:
                            self.metadata['species'].append(species)
                            self.metadata['audio_name'].append(audio.split('.')[0]+'.mp3')
                            self.metadata['spectrograms_name'].append(audio.split('.')[0]+f'_{c}.npy')
                            self.metadata['audio_offset'].append(float(line.strip().split('\t')[0]))
                            c+=1

        self._coder = preprocessing.LabelEncoder()
        self.metadata['species_encoded'] = self._coder.fit_transform(self.metadata['species'])

    def __len__(self):
        return len(self.metadata['audio_name'])

    def __getitem__(self, idx):
        ret = {}
        # general
        ret['species'] = self.metadata['species'][idx]
        ret['species_encoded'] = self.metadata['species_encoded'][idx]
        # audio
        path_audio = os.path.join(self.data_path, 'audio_files', self.metadata['species'][idx], self.metadata['audio_name'][idx])
        offset = self.metadata['audio_offset'][idx]
        wav, sr = librosa.load(path_audio, offset=offset, duration = 1.0)
        ret['audio'] = wav

        #spectograma
        path_spectogram = os.path.join(self.data_path, 'spectrograms', self.metadata['species'][idx], self.metadata['spectrograms_name'][idx])
        spectrogram = np.load(path_spectogram)
        ret['spectrogram'] = spectrogram

        return ret