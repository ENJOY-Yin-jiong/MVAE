import os
import torch
import pickle
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader

class BaseDataset(data.Dataset):
    def __init__(self, config):
        path_data_root = '../data/cp/ailab17k_from-scratch_cp'
        path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')
        path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')

        with open(path_dictionary, 'rb') as f:
            self.dictionary = pickle.load(f)
        event2word, word2event = self.dictionary

        self.n_class = []
        for key in event2word.keys():
            self.n_class.append(len(self.dictionary[0][key]))



        train_data = np.load(path_train_data)

        self.x = train_data['x']
        self.y = train_data['y']
        self.mask = train_data['mask']


        # self.n_group = train_data['num_groups']
        # print(train_data.keys())
        # print(self.n_group.shape)
        # num_batch = len(train_x) // batch_size

        # print('event2word', event2word)
        # print('word2event', word2event)
        # print('train_x', self.train_x.shape)

    # def split(self):

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # print(self.mask.shape)
        data_x = torch.from_numpy(self.x[idx, :, :])
        data_y = torch.from_numpy(self.y[idx, :, :])
        data_mask = torch.from_numpy(self.mask[idx, :])
        # data_group = self.n_group[idx]



        item = {
            'x': data_x,
            'y': data_y,
            'mask': data_mask,
            # 'group': data_group
        }

        return item



if __name__ == '__main__':
    dataset = BaseDataset()

    data = DataLoader(dataset,
                      batch_size=16,
                      shuffle=True,
                      num_workers=4,
                      pin_memory=False)

    for batch in data:
        print('x', batch['x'].shape)
        print('y', batch['y'].shape)
        print('mask', batch['mask'].shape)
        # print('group', batch['group'].shape)