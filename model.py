# model.py
# Complete VAE + Tokenizer with max_len fix

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import QED
import numpy as np

# ========================================
# 1. SMILES Tokenizer (with max_len)
# ========================================
class SMILESTokenizer:
    def __init__(self, max_len=120):
        self.max_len = max_len
        self.chars = ['<PAD>', '<START>', '<END>'] + sorted(list('CNOFSPClBrI#=+-()[]@'))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.chars)

    def encode(self, smiles):
        encoded = [self.char_to_idx['<START>']] + \
                  [self.char_to_idx.get(c, 0) for c in smiles] + \
                  [self.char_to_idx['<END>']]
        encoded = encoded[:self.max_len]
        padded = encoded + [self.char_to_idx['<PAD>']] * (self.max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long)

    def decode(self, indices):
        chars = []
        for idx in indices:
            if idx == self.char_to_idx['<END>']:
                break
            if idx != self.char_to_idx['<PAD>'] and idx != self.char_to_idx['<START>']:
                chars.append(self.idx_to_char.get(idx, ''))
        return ''.join(chars)

# ========================================
# 2. Conditional VAE (uses tokenizer.max_len)
# ========================================
class ConditionalVAE(nn.Module):
    def __init__(self, vocab_size, max_len=120, hidden_dim=256, latent_dim=64, condition_dim=1):
        super(ConditionalVAE, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Encoder
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + condition_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder_output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x, condition):
        embedded = self.embedding(x)
        _, (h_n, _) = self.encoder_lstm(embedded)
        h = h_n.squeeze(0)
        h_conditioned = torch.cat([h, condition], dim=1)
        mu = self.fc_mu(h_conditioned)
        logvar = self.fc_logvar(h_conditioned)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        z_conditioned = torch.cat([z, condition], dim=1)
        h = self.decoder_fc(z_conditioned).unsqueeze(1)  # (B, 1, H)
        h = h.repeat(1, self.max_len, 1)  # (B, max_len, H)
        lstm_out, _ = self.decoder_lstm(h)
        logits = self.decoder_output(lstm_out)
        return logits

    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, condition)
        return logits, mu, logvar

# ========================================
# 3. Loss + QED
# ========================================
def vae_loss(recon_logits, target, mu, logvar, beta=1.0):
    recon_loss = F.cross_entropy(recon_logits.view(-1, recon_logits.size(-1)), target.view(-1), ignore_index=0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

def compute_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    return QED.qed(mol)