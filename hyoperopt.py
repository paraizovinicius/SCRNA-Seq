import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import Dataset
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
import itertools


DATA_FOLDER = 'data'

df_info = pd.read_csv('data/GSE75688_final_sample_information.txt.gz', sep='\t')
df_info = df_info.rename(columns={'index': 'is_tumor'})


df = pd.read_csv('data/GSE75688_GEO_processed_Breast_Cancer_raw_TPM_matrix.txt.gz', sep='\t')

# Selecting columns representing cells
samples = df.loc[:,'BC01_02':]


# Renaming columns with gene names
columns = samples.T.columns.values
gene_name_list = list(df['gene_name'])
columns_mapping = dict(zip(columns, gene_name_list))
samples = samples.T.rename(columns=columns_mapping)

# Merging features (gene expressions) and labels (index = is_cancer)
labels = df_info.set_index(df_info['sample'])['is_tumor']
samples = samples.merge(labels.to_frame(), left_index=True, right_index=True)

def get_patient_id_for_each_sample(x):
    pattern = r'^BC(\d+)'
    indexes = x.index.values
    return [re.search(pattern, label).group(1) for label in indexes]

    
# Returns dataset before transformations
def get_dataset():
    x = samples.iloc[:,:-1]
    y = samples.is_tumor.map({'Tumor': 1, 'nonTumor': 0}) # reversing 0s and 1s to test how accuracy and f1 change if values are flipped
    patients = get_patient_id_for_each_sample(x)
    return x, y, patients


# Hyperparameter optimization for SDAE

x_original, y_original, patients = get_dataset()

scaler = StandardScaler() # Optimize: RobustScaler, StandardScaler, MinMaxScaler
scaler.fit(x_original)
X_scaled = scaler.transform(x_original)

class CustomDataset(Dataset):
    def __init__(self, data, labels, train=False, val=False):
        self.data = data
        self.labels = labels
        
        # Train/Val split (70%/30%)
        split_idx = int(0.70 * len(data))
        if train:
            self.data = data[:split_idx]
            self.labels = labels[:split_idx]
        elif val:
            self.data = data[split_idx:]
            self.labels = labels[split_idx:]
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_original.values, dtype=torch.long)

# transform dataset into a dataloader
ds_train = CustomDataset(x_tensor, y_tensor, train=True, val=False)
ds_val = CustomDataset(x_tensor, y_tensor, train=False, val=True)
dataset = CustomDataset(x_tensor, y_tensor, train=False, val=False)

# visualization of batches
dataloader = DataLoader(ds_train,batch_size=64,shuffle=False)


# Hyperparameters
input_layer = X_scaled.shape[1]
batch_size = 64
pretrain_epochs = 100 
finetune_epochs = 100

# Define hyperparameter search space
param_grid = {
    "hidden_layer": [10, 50, 100],
    "corruption": [0.1, 0.2],
    "lr_pretrain": [0.0003, 0.0002, 0.0001, 0.00005],
    "lr_train": [0.0003, 0.0002, 0.0001],
    "layers": [[300]],
    # "Scaler": [StandardScaler(), MinMaxScaler()]
    # "pacience_step_LR": [10,15,20],
}

# Utility: generate parameter combinations
def get_param_combinations(param_grid):
    keys = param_grid.keys()
    values = param_grid.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))
        

# Training wrapper
def train_sdae(params):
    input_layer = X_scaled.shape[1]
    hidden_layer = params["hidden_layer"]

    # Build network
    architecture = [input_layer] + params["layers"] + [hidden_layer]
    autoencoder = StackedDenoisingAutoEncoder(
        architecture, final_activation=None
    )

    print(f"\nTraining with params: {params}")

    # Pretraining
    ae.pretrain(
        ds_train,
        autoencoder,
        cuda=False,
        silent=True,
        validation=ds_val,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: torch.optim.Adam(model.parameters(), lr=params["lr_pretrain"]),
        scheduler=lambda x: ReduceLROnPlateau(x, mode='min', factor=0.8, patience=20, cooldown=5, threshold=0.0001),
        corruption=params["corruption"],
    )

    print("Training stage - Autoencoder.")
    ae_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=params['lr_train'])
    ae.train( 
        ds_train,
        autoencoder,
        cuda=False,
        silent=True,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=ReduceLROnPlateau(ae_optimizer, mode='min', factor=0.8, patience=20, cooldown=5, threshold=0.0001),
        corruption=params["corruption"],
    )
    # Metric for validation:
    losses = []
    for i in range(len(ds_train)):
        encoded = autoencoder.encoder(ds_train[i][0])
        decoded = autoencoder.decoder(encoded)
        mse = torch.nn.functional.mse_loss(decoded, ds_train[i][0])
        losses.append(mse.item())
    val_loss = np.mean(losses)
    print(f'MSE loss: {val_loss}')

    return val_loss, autoencoder


# Grid search loop
results = []
for params in get_param_combinations(param_grid):
    val_loss, model = train_sdae(params)
    results.append((params, val_loss))

# Sort results by best validation loss
results.sort(key=lambda x: x[1])

print("\nBest configuration:")
print(results[0])