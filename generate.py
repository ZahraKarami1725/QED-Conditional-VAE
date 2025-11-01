# generate.py
# Generate molecules using trained Conditional VAE
# Uses model.py and RDKit

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from model import SMILESTokenizer, ConditionalVAE, compute_qed
import os

# Load tokenizer and model
def load_model(model_path='models/vae.pth'):
    tokenizer = SMILESTokenizer(max_len=120)
    model = ConditionalVAE(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
        hidden_dim=256,
        latent_dim=64
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
    model.eval()
    return model, tokenizer

# Generate molecules
def generate_molecules(target_qed=0.6, num_samples=10):
    model, tokenizer = load_model()
    
    # Normalize QED (0-1)
    condition = torch.tensor([[target_qed]] * num_samples, dtype=torch.float)
    
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim)
        logits = model.decode(z, condition)
        samples = torch.argmax(logits, dim=-1)  # (B, max_len)
    
    generated_smiles = []
    valid_mols = []
    
    for i in range(num_samples):
        seq = samples[i].tolist()
        smiles = tokenizer.decode(seq)
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            qed = compute_qed(smiles)
            generated_smiles.append(smiles)
            valid_mols.append(mol)
            print(f"Valid | SMILES: {smiles} | QED: {qed:.3f}")
        else:
            print(f"Invalid SMILES: {smiles}")
    
    # Save image grid
    if valid_mols:
        img = Draw.MolsToGridImage(valid_mols, molsPerRow=5, subImgSize=(200, 200),
                                   legends=[f"QED: {compute_qed(s):.2f}" for s in generated_smiles])
        img.save('generated_molecules.png')
        print(f"\nImage saved: generated_molecules.png ({len(valid_mols)} valid molecules)")
    else:
        print("No valid molecules generated.")
    
    return generated_smiles

# Main
if __name__ == "__main__":
    print("Generating 10 molecules with target QED = 0.6")
    generate_molecules(target_qed=0.6, num_samples=10)