from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import numpy as np
from io import BytesIO
from PIL import Image

# Define the full molecule SMILES and substructure SMILES with weights
# full_smiles = 'OC(=O)CCC(O)=O.FC(F)(F)c1ccc2Sc3ccccc3N(CCCN4CCN(CCC5OCCCO5)CC4)c2c1'
# substructures = ['N1CCN(CC1)', 'CCCC', 'CCC5OCCCO5']
# weights = [0.2137, 0.1931, 0.1820]
# full_smiles = 'FC(F)(F)c1ccc2Sc3ccccc3N(CCCN4CCN(CC4)C5CC5)c2c1'
# substructures = ['c1ccc2Sc3ccccc3Nc2c1', 'C5CC5', 'N4CCN(CC4)']
# weights = [0.2333, 0.2025, 0.2149]
full_smiles = 'CN1CCN(CC1)CC(=O)N2c3ccccc3C(=O)Nc4cccnc24'
substructures = ['CC(=O)N', 'N1CCN(CC1)']
weights = [0.3366, 0.4519]

# Create RDKit molecule objects
full_mol = Chem.MolFromSmiles(full_smiles)
sub_mols = [Chem.MolFromSmarts(s) for s in substructures]

# Calculate 2D coordinates for the molecules
rdDepictor.Compute2DCoords(full_mol)

# Normalize weights to a range suitable for coloring (0-1)
norm_weights = (np.array(weights) - min(weights)) / (max(weights) - min(weights))
colors = [(w, 0, 1 - w) for w in norm_weights]  # Interpolate between blue and red

# Prepare the drawer
opts = rdMolDraw2D.MolDrawOptions()
opts.useBWAtomPalette()
opts.atomLabelFontSize = 16
opts.bondLineWidth = 3
drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)  # Use Cairo for PNG
drawer.SetDrawOptions(opts)

# Draw the full molecule
drawer.DrawMolecule(full_mol)

# Highlight the substructures
for sub_mol, color in zip(sub_mols, colors):
    matches = full_mol.GetSubstructMatches(sub_mol)
    print(matches)
    for match in matches:
        drawer.DrawMolecule(
            full_mol,
            highlightAtoms=match,
            highlightBonds=[full_mol.GetBondBetweenAtoms(match[i], match[j]).GetIdx()
                            for i in range(len(match)) for j in range(i + 1, len(match))
                            if full_mol.GetBondBetweenAtoms(match[i], match[j])],
            highlightAtomColors={atom: color for atom in match}
        )

# Finish the drawing
drawer.FinishDrawing()

# Save the image to a byte stream
png = drawer.GetDrawingText()

# Convert the byte stream to an image object
img = Image.open(BytesIO(png))

# Save the image to a file
# filename = 'MILI_1.png'
# filename = 'MILI_2.png'
filename = 'MILI_3.png'
img.save(filename)
