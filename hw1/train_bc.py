import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from policy import PolicyNet
from tqdm import tqdm
from utils import set_seed

DATA_PATH = "data/expert_dataset_10.csv"
MODEL_SAVE = "models/model_bc_10.pth"
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
SEED = 33

set_seed(SEED)

df = pd.read_csv(DATA_PATH)
X = torch.tensor(df[["x", "x_dot", "theta", "theta_dot"]].values, dtype=torch.float32)
y = torch.tensor(df["action"].values, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

obs_dim = X.shape[1]
policy = PolicyNet(obs_dim, action_dim=2, hidden_dim=64)
optimizer = optim.AdamW(policy.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(EPOCHS)):
    total_loss = 0
    for xb, yb in loader:
        logits = policy(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    print(f"Epoch {epoch + 1}/{EPOCHS}, loss={total_loss/len(dataset)}")

torch.save(policy.state_dict(), MODEL_SAVE)
