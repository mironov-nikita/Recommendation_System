import torch 
from torch import nn 
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

from Model import TwoTowersModel
from DataPreparation import get_train_and_test_datasets, leave_one_out_split, load_ratings

checkpoint_folder_path = Path(__file__).parent / "checkpoints"
checkpoint_folder_path.mkdir(exist_ok=True)
checkpoint_path = Path(__file__).parent / "checkpoints" / "checkpoint1.pt"

N_USERS = 2000
_ratings = load_ratings()
N_ITEMS = _ratings["item"].max() + 1
print(N_ITEMS)

def train(epochs=3):
    model.train()

    for epoch in range(1, epochs+1):
        epoch_loss = 0.0

        for users, items, ratings in train_loader:

            optimizer.zero_grad()

            logits = model(users, items)
            loss = loss_fn(logits, ratings)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch count | {epoch}/{epochs}")
        print(f"LOSS | {epoch_loss / len(train_loader):.6f}")
    torch.save(model.state_dict(), checkpoint_path)
    print("Weights were saved")


def test():
    model.eval()
    with torch.no_grad():
        loss = 0.0

        for users, items, ratings in test_loader:
            pred = model(users, items)
            loss += loss_fn(pred, ratings).item()

    return loss



if __name__ == "__main__":

    train_dataset, test_dataset = get_train_and_test_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False
    )
    
    model = TwoTowersModel(N_USERS, N_ITEMS)

    if checkpoint_path.exists():
        
        model.load_state_dict(torch.load(checkpoint_path))
        #print(model.parameters())
    else:
        print("Weights not found")

    optimizer = torch.optim.Adam(lr=0.005, params=model.parameters())
    loss_fn = nn.MSELoss()

    train()
    print(test())
