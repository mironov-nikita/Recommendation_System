import torch 
from torch import nn 
import pandas as pd

from Data import leave_one_out_split, load_ratings


class MFDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame):
        self.users = torch.tensor(data["user"], dtype=torch.float32)
        self.items = torch.tensor(data["item"], dtype=torch.float32)
        self.ratings = torch.tensor(data["ratings"], dtype=torch.float32)


    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, index):
        return self.users[index], self.items[index], self.ratings[index]
    
if __name__ == "__main__":
    train, test = leave_one_out_split(load_ratings())

    train_data = MFDataset(train)
    print(len(train_data))
    print(train_data[0])

    loader = torch.utils.data.DataLoader(train_data, batch_size=4)
    users, items, ratings = next(iter(loader))
    print(users.shape, users.dtype)
    