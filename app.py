# app.py
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from model import SMILESTokenizer, ConditionalVAE, compute_qed

@st.cache_resource
def load_model():
    tokenizer = SMILESTokenizer(max_len=120)
    model = ConditionalVAE(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.max_len,
        hidden_dim=256,
        latent_dim=64
    )
    model.load_state_dict(torch.load('models/vae.pth', map_location='cpu'))
    model.eval()
    return model, tokenizer

st.set_page_config(page_title="Molecule Generator", layout="centered")
st.title("QED-Conditioned Molecule Generator")

try:
    model, tokenizer = load_model()
    st.success("Model loaded!")
except Exception as e:
    st.error(f"Run `train.py` first!\n{e}")
    st.stop()

target_qed = st.slider("Target QED", 0.0, 1.0, 0.6, 0.05)

if st.button("Generate Molecule", type="primary"):
    with st.spinner("Generating..."):
        condition = torch.tensor([[target_qed]])
        with torch.no_grad():
            z = torch.randn(1, model.latent_dim)
            logits = model.decode(z, condition)
            seq = torch.argmax(logits, dim=-1)[0]
            smiles = tokenizer.decode(seq.tolist())
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            qed_actual = compute_qed(smiles)
            img = Draw.MolToImage(mol, size=(400, 400))
            
            # Fixed: use_container_width instead of use_column_width
            st.image(img, use_container_width=True)
            
            st.code(smiles, language="text")
            st.metric("Actual QED", f"{qed_actual:.3f}")
            if abs(qed_actual - target_qed) < 0.1:
                st.success("QED close to target!")
            else:
                st.warning("QED slightly off — model is learning!")
        else:
            st.error("Invalid SMILES. Try again!")