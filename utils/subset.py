import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def reduce_dataset_size(dataset, max_samples,random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    subset_indices = torch.randperm(len(dataset))[:max_samples]
    reduced_dataset = Subset(dataset, subset_indices)
    return reduced_dataset