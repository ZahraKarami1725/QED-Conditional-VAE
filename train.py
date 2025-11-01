# train.py
# Complete, fixed training script: QM9 → SMILES → Conditional VAE
# Auto-download, fallback dataset, max_len fix, low RAM

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch_geometric.datasets as datasets
import pandas as pd
import numpy as np
import requests
from io import StringIO
from model import SMILESTokenizer, ConditionalVAE, vae_loss, compute_qed
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem

# ========================================
# 1. Convert QM9 graph to SMILES (robust)
# ========================================
def graph_to_smiles(data):
    try:
        rwmol = Chem.RWMol()

        # Add atoms from z (atomic numbers)
        atom_map = {}
        for i, z_val in enumerate(data.z):
            atom = Chem.Atom(int(z_val.item()))
            idx = rwmol.AddAtom(atom)
            atom_map[i] = idx

        # Bond types: 1=single, 2=double, 3=triple, 4=aromatic
        bond_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC
        }

        # Add bonds
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            tgt = edge_index[1, i].item()
            bond_order = int(edge_attr[i].item()) if edge_attr is not None else 1
            bond_type = bond_types.get(bond_order, Chem.BondType.SINGLE)
            rwmol.AddBond(atom_map[src], atom_map[tgt], bond_type)

        # Add hydrogens (critical!)
        rwmol = Chem.AddHs(rwmol)

        # Sanitize with kekulization
        Chem.SanitizeMol(rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.Kekulize(rwmol, clearAromaticFlags=True)

        # Generate SMILES
        smiles = Chem.MolToSmiles(rwmol, isomericSmiles=False, canonical=True)
        if not smiles:
            smiles = Chem.MolToSmiles(rwmol, kekuleSmiles=True)

        return smiles if len(smiles) > 1 else None

    except Exception as e:
        # print(f"Debug: {e}")
        return None


# ========================================
# 2. Fallback: Download ZINC SMILES
# ========================================
def download_zinc_subset(num_samples=1000):
    print("Using ZINC subset as fallback...")
    try:
        url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_smiles.csv"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), header=None, names=['smiles'])
            smiles = df['smiles'].head(num_samples).tolist()
            print(f"Downloaded {len(smiles)} ZINC SMILES.")
            return smiles
    except:
        pass

    # Hardcoded fallback
    hardcoded = [
        'CCO', 'CC(=O)O', 'c1ccccc1', 'CC1=CC=CC=C1', 'CCOC(=O)C1=CC=CC=C1',
        'NC1=CC=CC=C1', 'OC1=CC=CC=C1', 'CCN(CC)C(=O)C1=CC=CC=C1', 'C1CCOC1',
        'CN1C=NC2=C1C(=O)N(C)C(=O)N2C', 'CC(C)CC(=O)O', 'CC(C)N'
    ]
    smiles = (hardcoded * ((num_samples // len(hardcoded)) + 1))[:num_samples]
    print(f"Using {len(smiles)} hardcoded SMILES.")
    return smiles


# ========================================
# 3. Prepare data (QM9 or fallback)
# ========================================
def prepare_data(subset_size=100):
    csv_path = 'data/qm9_processed.csv'
    if os.path.exists(csv_path):
        print("Loading preprocessed data...")
        return pd.read_csv(csv_path)

    print("Preparing dataset...")
    smiles_list = []

    # Try QM9
    try:
        dataset = datasets.QM9(root='data/')
        for i, data in enumerate(dataset):
            if i >= subset_size:
                break
            smiles = graph_to_smiles(data)
            if smiles:
                smiles_list.append(smiles)
            if i % 20 == 0:
                print(f"  QM9: {i} processed, {len(smiles_list)} valid")

        if len(smiles_list) >= 10:
            print(f"QM9 success: {len(smiles_list)} SMILES.")
        else:
            raise ValueError("Too few QM9 SMILES.")
    except Exception as e:
        print(f"QM9 failed: {e}. Using fallback.")
        smiles_list = download_zinc_subset(subset_size)

    # Compute QED
    qed_scores = [compute_qed(s) for s in smiles_list]
    scaler = MinMaxScaler()
    qed_normalized = scaler.fit_transform(np.array(qed_scores).reshape(-1, 1)).flatten()

    df = pd.DataFrame({'smiles': smiles_list, 'qed_norm': qed_normalized})
    os.makedirs('data', exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Data saved: {len(df)} samples.")
    return df


# ========================================
# 4. Dataset class
# ========================================
class SMILESDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.smiles = df['smiles'].values
        self.qed = df['qed_norm'].values
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.smiles[idx])
        qed_cond = torch.tensor([self.qed[idx]], dtype=torch.float)
        return encoded, qed_cond


# ========================================
# 5. Main training
# ========================================
if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Tokenizer with max_len
    tokenizer = SMILESTokenizer(max_len=120)

    # Prepare data
    df = prepare_data(subset_size=100)

    # Dataset + Loader
    dataset = SMILESDataset(df, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    # Model with max_len
    model = ConditionalVAE(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
        hidden_dim=256,
        latent_dim=64
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("Starting training...")
    for epoch in range(5):
        model.train()
        total_loss = 0.0
        for smiles_batch, condition_batch in dataloader:
            optimizer.zero_grad()
            logits, mu, logvar = model(smiles_batch, condition_batch)
            loss = vae_loss(logits, smiles_batch, mu, logvar, beta=0.1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")

    # Save
    torch.save(model.state_dict(), 'models/vae.pth')
    print("Training complete! Model saved.")
    print("Next: python generate.py or streamlit run app.py")