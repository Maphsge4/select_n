import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, length, batch_size, vocab_size, seq_len=128):
        self.len = length
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.data = torch.randint(3, self.vocab_size, (length, seq_len))  # vocab_size: 30522

    def __getitem__(self, index):
        # return self.data[:, :, :, index]
        input_ids = self.data[index]
        mask_labels = torch.randint(3, self.vocab_size, (int(self.seq_len*0.15),))  # vocab_size: 30522
        return (input_ids, mask_labels)

    def __len__(self):
        return self.len
    
        
def prepare_dataloader(length, batch_size, vocab_size):
    # train_sampler = ElasticDistributedSampler(train_dataset)
    return DataLoader(
        RandomDataset(length, batch_size, vocab_size),
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        # sampler=train_sampler,
    )