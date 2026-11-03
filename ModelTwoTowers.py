import torch 
from torch import nn
from torch.utils.data import DataLoader


class UsersTower(nn.Module):
    def __init__(self, users_number):
        super().__init__()
        self.embeddings = nn.Embedding(users_number, 128)

    def forward(self, x):
        vec = self.embeddings(x)
        return vec
    

class ItemsTower(nn.Module):
    def __init__(self, items_number):
        super().__init__()
        self.embeddings = nn.Embedding(items_number, 128)

    def forward(self, x):
        vec = self.embeddings(x)
        return vec


class TwoTowersModel(nn.Module):
    def __init__(self, users_number, items_number):
        super().__init__()
        self.users_emb = UsersTower(users_number)
        self.items_emb = ItemsTower(items_number)

    def forward(self, users_idx, items_idx):
        users_vec = self.users_emb(users_idx) 
        items_vec = self.items_emb(items_idx)
        logits = (users_vec * items_vec).sum(-1)
        return logits 

    def score_for_all(self, user_idx):
        items_number = self.items_emb.embeddings.num_embeddings
        all_items = torch.arange(items_number)
        Q = self.items_emb(all_items) 
        scores = self.users_emb(user_idx) @ Q.T
        return scores

if __name__ == "__main__":
    model = TwoTowersModel(10, 10)
    print(model.score_for_all(torch.tensor([1])))