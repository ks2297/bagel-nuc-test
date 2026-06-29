"""
Key amino acid constants

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

# {1-letter: 3-letter} aminoacid names
aa_dict = {
    'A': 'ALA',  # alanine
    'R': 'ARG',  # arginine
    'N': 'ASN',  # asparagine
    'D': 'ASP',  # aspartic acid
    'C': 'CYS',  # cysteine (not mutatable by default)
    'Q': 'GLN',  # glutamine
    'E': 'GLU',  # glutamic acid
    'G': 'GLY',  # glycine
    'H': 'HIS',  # histidine
    'I': 'ILE',  # isoleucine
    'L': 'LEU',  # leucine
    'K': 'LYS',  # lysine
    'M': 'MET',  # methionine
    'F': 'PHE',  # phenylalanine
    'P': 'PRO',  # proline
    'S': 'SER',  # serine
    'T': 'THR',  # threonine
    'W': 'TRP',  # tryptophan
    'Y': 'TYR',  # tyrosine
    'V': 'VAL',  # valine
}

aminoacids_letters = list(aa_dict.keys())

# {1-letter: 3-letter} nucleotide names, following the PDB Chemical Component Dictionary (CCD).
# DNA deoxyribonucleotides are prefixed with 'D'; RNA ribonucleotides use the bare base letter.
dna_dict = {
    'A': 'DA',  # deoxyadenosine
    'C': 'DC',  # deoxycytidine
    'G': 'DG',  # deoxyguanosine
    'T': 'DT',  # deoxythymidine
}

rna_dict = {
    'A': 'A',  # adenosine
    'C': 'C',  # cytidine
    'G': 'G',  # guanosine
    'U': 'U',  # uridine
}

# Registry of per-molecule-type alphabets. The 'protein' entry IS aa_dict (same object),
# so the protein path validates and maps three-letter names byte-identically to before.
# molecule_type tags on Residue/Chain index into this registry to route validation.
alphabets = {
    'protein': aa_dict,
    'dna': dna_dict,
    'rna': rna_dict,
}

mutation_bias = {aa: 1.0 / len(aa_dict) for aa in aa_dict.keys()}

mutation_bias_no_cystein = {aa: 1.0 / (len(aa_dict) - 1) if aa != 'C' else 0.0 for aa in aa_dict.keys()}

# Per-molecule-type substitution biases, uniform within each nucleotide alphabet.
# Mirrors the `alphabets` registry: mutation protocols index this by a chain's molecule_type.
# The 'protein' entry is the existing no-cysteine default, so protein sampling is unchanged.
dna_mutation_bias = {base: 1.0 / len(dna_dict) for base in dna_dict.keys()}
rna_mutation_bias = {base: 1.0 / len(rna_dict) for base in rna_dict.keys()}

mutation_biases = {
    'protein': mutation_bias_no_cystein,
    'dna': dna_mutation_bias,
    'rna': rna_mutation_bias,
}

hydrophobic_residues = ('VAL', 'ILE', 'LEU', 'PHE', 'MET', 'TRP')

backbone_atoms = ('CA', 'N', 'C')  # protein backbone (kept for backwards compatibility)

# Per-molecule-type backbone atom names, used by geometric energy terms (centroid/symmetry/etc).
# Nucleic-acid backbone follows the phosphodiester trace; the exact atom names should be
# confirmed against the oracle's AtomArray (step-6 spike, Q2) and tuned if needed.
protein_backbone_atoms = ('CA', 'N', 'C')
nucleic_backbone_atoms = ('P', "O5'", "C5'", "C4'", "C3'", "O3'")

# Union over all polymer types. Because protein and nucleic-acid backbone atom names are
# disjoint, masking a structure with this union selects each residue's correct backbone atoms,
# regardless of type -- so protein-only structures behave exactly as with `backbone_atoms`,
# and nucleic-acid (or mixed) structures no longer mask to an empty set.
all_backbone_atoms = tuple(dict.fromkeys(protein_backbone_atoms + nucleic_backbone_atoms))

angstrom = 1.0  # Units of measure for distances
nm = 10.0  # nm value in units of measure for distances

# These are the maximum values calculated for different atom types using a probe radius of 1.4 Å,
# GPT suggests Lee&Richards 1971 and Connolly as references.
max_sasa_values = {
    'H': 14.0 * angstrom**2,
    'C': 20.0 * angstrom**2,
    'N': 16.0 * angstrom**2,
    'O': 17.0 * angstrom**2,
    'S': 22.0 * angstrom**2,
    'P': 24.0 * angstrom**2,
}

probe_radius_water = 1.4 * angstrom

# From OpenFold, used in ESMFold
# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    'N',
    'CA',
    'C',
    'CB',
    'O',
    'CG',
    'CG1',
    'CG2',
    'OG',
    'OG1',
    'SG',
    'CD',
    'CD1',
    'CD2',
    'ND1',
    'ND2',
    'OD1',
    'OD2',
    'SD',
    'CE',
    'CE1',
    'CE2',
    'CE3',
    'NE',
    'NE1',
    'NE2',
    'OE1',
    'OE2',
    'CH2',
    'NH1',
    'NH2',
    'OH',
    'CZ',
    'CZ2',
    'CZ3',
    'NZ',
    'OXT',
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

# This dictionary contains hydrophobicity values for each amino acid, based on the GRAVY (Grand Average of Hydropathy) score.
# The values are taken from the Kyte-Doolittle scale, which is commonly used to assess the hydrophobicity of amino acids.
hydropathy_index = {
    'ILE': 4.5,  # Isoleucine
    'VAL': 4.2,  # Valine
    'LEU': 3.8,  # Leucine
    'PHE': 2.8,  # Phenylalanine
    'CYS': 2.5,  # Cysteine
    'MET': 1.9,  # Methionine
    'ALA': 1.8,  # Alanine
    'GLY': -0.4,  # Glycine
    'THR': -0.7,  # Threonine
    'TRP': -0.9,  # Tryptophan
    'SER': -0.8,  # Serine
    'TYR': -1.3,  # Tyrosine
    'PRO': -1.6,  # Proline
    'HIS': -3.2,  # Histidine
    'GLU': -3.5,  # Glutamic acid
    'GLN': -3.5,  # Glutamine
    'ASP': -3.5,  # Aspartic acid
    'ASN': -3.5,  # Asparagine
    'LYS': -3.9,  # Lysine
    'ARG': -4.5,  # Arginine
}

# This maximum value possible for a residue calculated theoretically, could be used for normalization
# The value is based on the largest amino acid (Tryptophan) and assumes all its atoms are fully exposed to solvent,
# which is a theoretical upper bound for SASA of a residue.
# Ref: Tein et al, 2013 https://pmc.ncbi.nlm.nih.gov/articles/PMC3836772/ for max SASA values of amino acids
max_residue_sasa = 285.0 * angstrom**2

# This dictionary contains the maximum theoretical SASA values for each amino acid, used for normalization.
max_theoretical_sasa_for_residues = {
    'TRP': 285.0 * angstrom**2,  # Tryptophan
    'ARG': 274.0 * angstrom**2,  # Arginine
    'TYR': 263.0 * angstrom**2,  # Tyrosine
    'PHE': 240.0 * angstrom**2,  # Phenylalanine
    'LYS': 236.0 * angstrom**2,  # Lysine
    'GLN': 225.0 * angstrom**2,  # Glutamine
    'HIS': 224.0 * angstrom**2,  # Histidine
    'MET': 224.0 * angstrom**2,  # Methionine
    'GLU': 223.0 * angstrom**2,  # Glutamate
    'LEU': 201.0 * angstrom**2,  # Leucine
    'ILE': 197.0 * angstrom**2,  # Isoleucine
    'ASN': 195.0 * angstrom**2,  # Asparagine
    'ASP': 193.0 * angstrom**2,  # Aspartate
    'VAL': 174.0 * angstrom**2,  # Valine
    'THR': 172.0 * angstrom**2,  # Threonine
    'CYS': 167.0 * angstrom**2,  # Cysteine
    'PRO': 159.0 * angstrom**2,  # Proline
    'SER': 155.0 * angstrom**2,  # Serine
    'ALA': 129.0 * angstrom**2,  # Alanine
    'GLY': 104.0 * angstrom**2,  # Glycine
}
