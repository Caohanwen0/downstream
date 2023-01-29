import torch
import torch.utils.data as data
from model_center.dataset import MMapIndexedDataset
import random
import numpy as np
import scipy.linalg

class RobertaDataset(data.Dataset):
    def __init__(self, input_ids:     MMapIndexedDataset,
                       attention_mask:        MMapIndexedDataset,):

        self.input_ids = input_ids
        self.input_ids = input_ids,
        self.attention_mask = attention_mask
   
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = np.array(self.input_ids[index], dtype='int32')
        lm_pos = self.lm_pos[index]
        masked_labels = self.masked_labels[index]

        input_length = len(input_ids)
        if input_length < self.max_seq_length: # padding to max length
            input_ids = np.pad(input_ids, (0, self.max_seq_length - input_length))

        ones = np.ones([input_length, input_length])
        zeros = np.zeros([self.max_seq_length - input_length, self.max_seq_length - input_length])
        attention_mask = scipy.linalg.block_diag(ones, zeros)

        labels = np.full([self.max_seq_length, ], -100, dtype="int32")
        labels[lm_pos] = masked_labels
        return torch.LongTensor(input_ids), torch.LongTensor(attention_mask).byte(), torch.LongTensor(labels)
 