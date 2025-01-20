
import numpy as np
import torch

def noise_data_reduction(features, labels, data_dict = {0:True, 1:False, 2:False, 3:False, 4:False, 5:False, 6:False}, noise_to_data_ratio = 1):
    
    
    data_size = 0
    for key, val in data_dict.items():
        if val == 1:
            data_size += (labels == key).sum()
            labels[labels == key] = 0
    noise_size = int(data_size * noise_to_data_ratio)
    
    mask = torch.zeros_like(labels)
    
    for key, val in data_dict.items():
        if val  == 0:
            mask += (labels == key)
            labels[labels == key] = 1
    
    
    
    
    noise_indexes = np.where(labels == 1)
    data_indexes = np.where(labels == 0)
    # print(noise_indexes)
    data_size = len(data_indexes[0])
    noise_size = len(noise_indexes[0])
    new_noise_size = int(data_size * noise_to_data_ratio)
    new_noise_indexes = np.random.permutation(noise_indexes[0])[:new_noise_size]
    data_labels = labels[data_indexes]
    noise_labels = labels[new_noise_indexes]
    data_features = features[data_indexes]  
    noise_features = features[new_noise_indexes]
    new_features = torch.cat((data_features, noise_features), dim = 0)
    new_labels = torch.cat((data_labels, noise_labels), dim = 0)
    
    
    
    # new_data = torch.cat((features[data_indexes], features[noise_indexes]), dim = 0)
    # new_labels = torch.cat((labels[data_indexes], labels[noise_mask]), dim = 0)
    
    return new_features, new_labels
    
    
def read_in_data(directory : str, conversion_dict : dict[int,int] = {0:0,1:1,2:2,3:3,4:4,5:5,6:6}):
    data = np.load(directory)
    features =  data[:,:-1]
    labels = data[:,-1]
    for original, new in conversion_dict.items():
        labels[labels == original] = new
    return torch.tensor(features).float(), torch.tensor(labels).long()

