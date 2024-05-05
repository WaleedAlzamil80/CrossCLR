import os
import argparse

import numpy as np 
import torch

from torchvision import datasets
from torchvision.transforms import v2

from helpful.Sampling_techniques import *
from helpful.losses import *
from helpful.Models import *

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if cuda else torch.device('cpu')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

BASE_DIR = os.getcwd()

# Define command-line arguments
parser = argparse.ArgumentParser(description="Specify the training parameters.")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size (default: 1024)")
parser.add_argument("--epochs", type=int, default=15, help="Epochs (default: 15)")
parser.add_argument("--N", type=int, default=16, help="Negative samples prt instance per patch (default: 16)")

args = parser.parse_args()

trans = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(25),
    v2.RandomAffine(degrees=0, translate=(0, 0.2), scale=(1.0, 1.2), shear=20),
    v2.Normalize(mean=[0.5], std=[0.5])
])

if cuda:
    torch.cuda.empty_cache()

model = EncoderMinist().to(device)

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=trans, # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

test_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=False, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale = True)]) # images come as PIL format, we want to turn into Torch tensors
)

train_generator = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
test_generator = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle = False)


# inistantiate your optimizer and the scheduler procedure
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [15, 35], gamma = 0.5)

# Define an empty lists to save the losses during the training process
train_loss_values = []
var_loss_values = []
cov_loss_values = []
con_loss_values = []


for epoch in range(args.epochs):

    # Training Mode
    model.train()
    train_loss = 0
    var_lossv = 0
    con_lossv = 0
    cov_lossv = 0


    for train_x, train_y in train_generator:

        pos, neg = sample_contrastive_pairs_SL(train_x, train_y, args.N)
        train_x, pos, neg = train_x.to(device), pos.to(device), neg.to(device)

        # Feedforward
        emb_actu = model(train_x)
        emb_pos = model(pos)
        emb_neg = model(neg.reshape(-1, 1, 28, 28)).reshape(-1, args.N, 512)
        _std_loss = std_loss(emb_actu, emb_pos)
        _cov_loss = cov_loss(emb_actu, emb_pos)
        loss_contra = contrastive_loss(emb_actu, emb_pos, emb_neg)

        loss = (loss_contra + 2*_std_loss + _cov_loss) / 4
        train_loss += loss.item()
        con_lossv += loss_contra.item()
        var_lossv += _std_loss.item()
        cov_lossv += _cov_loss.item()


        # At start of each Epoch
        optimizer.zero_grad()

        # Do the back probagation and update the parameters
        loss.backward()
        optimizer.step()

    train_loss /= len(train_generator)
    con_lossv /= len(train_generator)
    cov_lossv /= len(train_generator)
    var_lossv /= len(train_generator)

    scheduler.step()

    train_loss_values.append(train_loss)
    var_loss_values.append(var_lossv)
    cov_loss_values.append(cov_lossv)
    con_loss_values.append(con_lossv)

    print(f"Epoch: {epoch + 1} | training_loss : {train_loss:.4f} | Sim_loss : {loss_contra:.4f} | Var_loss : {_std_loss:.4f} | Cov_loss : {_cov_loss:.4f}")


# Evaluation mode
model.eval()
embeddings = []
with torch.inference_mode():
  for test_x, test_y in test_generator:
      # Feedforward again for the evaluation phase
      embedding = model(model.encoder(test_x.to(device)).reshape((-1, 128 * 28 * 28)))
      embeddings.append(embedding.cpu().detach().numpy())

# plt.figure(figsize=(8, 8))
# for embed in embeddings:
#   plt.scatter(embed[:, 0], embed[:, 1], c="black", alpha=0.5, s=3)
# plt.show()

# Convert lists to NumPy arrays
train_loss_values = np.array(train_loss_values)
var_loss_values = np.array(var_loss_values)
cov_loss_values = np.array(cov_loss_values)
con_loss_values = np.array(con_loss_values)

# Create the new folder
directory = os.path.join(BASE_DIR, "results")
os.makedirs(directory, exist_ok=True)

# Save NumPy arrays with specified directory
np.save(os.path.join(directory, 'train_loss_values.npy'), train_loss_values)
np.save(os.path.join(directory, 'var_loss_values.npy'), var_loss_values)
np.save(os.path.join(directory, 'cov_loss_values.npy'), cov_loss_values)
np.save(os.path.join(directory, 'con_loss_values.npy'), con_loss_values)
# Save the embeddings to a .npy file
np.save(os.path.join(directory, 'Embedding.npy'), np.array(embeddings))

# Specify the file name for saving the model
model_file = os.path.join(directory, 'model.pt')

# Save the model to the specified directory
torch.save(model.state_dict(), model_file)