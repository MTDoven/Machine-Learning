import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset

# 10000 items for training, 10000 items for valid, of all 20000 items
Num_train = 50

# load data
train_data = CriteoDataset('./data', train=True)
loader_train = DataLoader(train_data, batch_size=200,
                          sampler=sampler.SubsetRandomSampler(range(Num_train)))
val_data = CriteoDataset('./data', train=True)
loader_val = DataLoader(val_data, batch_size=200,
                        sampler=sampler.SubsetRandomSampler(range(Num_train, 100)))

feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print(feature_sizes)

epochs = 1000
model = DeepFM(feature_sizes, use_cuda=True, embedding_size=256, hidden_dims=[256, 256, 256], dropout=[0.2, 0.2, 0.2])
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
model.fit(loader_train, loader_val, optimizer, scheduler, epochs=epochs, verbose=True, print_every=1000)
