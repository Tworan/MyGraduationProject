import random
import numpy as np

class Transform(object):
    def __init__(self, processes):
        '''
        processes: should be list
        '''
        self.processes = processes
    
    def __call__(self, sample):
        for process in self.processes:
            sample = process(sample)
        
        return sample
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std 
    
    def __call__(self, sample):
        '''
        input: sample should be [C, H, W]
        '''
        sample = (sample - self.mean) / (self.std + 1e-7)
        return sample
    
class CenterCrop(object):
    def __init__(self, size):
        '''
        input: size should be a tuple - [cH, cW]
        '''
        self.size = size

    def __call__(self, sample):
        '''
        input: sample should be [C, H, W]
        '''
        C, H, W = sample.shape
        cH, cW = self.size
        edgeH, edgeW = int((H - cH) // 2), int((W - cW) // 2)
        sample = sample[:, edgeH: edgeH+cH, edgeW: edgeW+cW]
        return sample

class RandomCrop(object):
    def __init__(self, size):
        '''
        input: size should be a tuple - [cH, cW]
        '''
        self.size = size

    def __call__(self, sample):
        '''
        input: sample should be [C, H, W]
        '''
        C, H, W = sample.shape
        cH, cW = self.size
        edgeH, edgeW = random.randint(0, H-cH), random.randint(0, W-cW)
        sample = sample[:, edgeH: edgeH+cH, edgeW: edgeW+cW]
        return sample
    
class HorizontalFlip(object):
    def __init__(self, filp_ratio):
        self.flip_ratio = filp_ratio
    
    def __call__(self, sample):
        if random.random() < self.flip_ratio:
            sample = sample[:, :, ::-1]
        return sample 

def get_preprocessing_pipeline():
    pipelines = {}
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    pipelines['train'] = Transform([
        Normalize(0, 255.),
        RandomCrop(crop_size),
        HorizontalFlip(0.5),
        Normalize(mean, std)
    ])
    pipelines['test'] = Transform([
        Normalize(0, 255.),
        CenterCrop(crop_size),
        Normalize(mean, std)
    ])
    pipelines['val'] = pipelines['test']
    return pipelines