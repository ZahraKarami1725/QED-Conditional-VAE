# make_demo.py
# Creates demo.gif for README.md
import imageio
import numpy as np
from rdkit.Chem import Draw, MolFromSmiles
from PIL import Image, ImageDraw, ImageFont

# Use generate.py to get real molecules
from generate import generate_molecules

print("Generating molecules for demo...")
smiles_list = generate_molecules(target_qed=0.6, num_samples=5)

# Create frames
images = []
for i, smiles in enumerate(smiles_list):
    mol = MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        pil_img = Image.fromarray(np.array(img))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), f"QED Target: 0.6", fill="red", font=font)
        draw.text((10, 260), f"#{i+1}", fill="white", font=font)
        images.append(np.array(pil_img))

# Save GIF
if images:
    imageio.mimsave('demo.gif', images, fps=1, loop=0)
    print("demo.gif created! Add to README.")
else:
    print("No images to save.")