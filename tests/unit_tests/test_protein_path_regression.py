"""Regression check: the protein path must be byte-identical after DNA/RNA support.

These assertions need no oracle and no GPU (run with `pytest --oracles skip`).
They lock down the invariants that guarantee existing protein behaviour did not
move: the protein alphabet, mutation bias, three-letter mapping, and backbone
masking are all exactly what they were before the nucleic-acid work.

The live end-to-end protein regression (a full ESMFold design run reproducing a
known trajectory) needs a GPU and is run separately.
"""

import numpy as np
import biotite.structure as struc

import bagel as bg
from bagel import constants as C


def test_protein_alphabet_is_same_object() -> None:
    assert C.ALPHABETS['protein'] is C.aa_dict


def test_protein_mutation_bias_unchanged() -> None:
    assert C.MUTATION_BIASES['protein'] is C.mutation_bias_no_cystein
    assert C.mutation_bias == {aa: 1.0 / 20 for aa in C.aa_dict}
    # no-cysteine default: C has zero probability, others equal
    assert C.mutation_bias_no_cystein['C'] == 0.0
    assert abs(sum(C.mutation_bias_no_cystein.values()) - 1.0) < 1e-9


def test_protein_three_letter_mapping_unchanged() -> None:
    for one, three in [('A', 'ALA'), ('G', 'GLY'), ('W', 'TRP')]:
        assert bg.Residue(one, 'X', 0).three_letter_name == three


def test_protein_backbone_mask_unchanged() -> None:
    prot = struc.array(
        [
            struc.Atom([0, 0, 0], chain_id='A', res_id=1, res_name='ALA', atom_name=n, element='C')
            for n in ('N', 'CA', 'C', 'CB', 'O')
        ]
    )
    assert np.array_equal(
        np.isin(prot.atom_name, ('CA', 'N', 'C')),
        np.isin(prot.atom_name, C.all_backbone_atoms),
    )


def test_protein_chain_construction_unchanged() -> None:
    chain = bg.Chain([bg.Residue(aa, 'A', i) for i, aa in enumerate('ACDEFG')])
    assert chain.molecule_type == 'protein'
    assert chain.sequence == 'ACDEFG'
    chain.mutate_residue(0, 'W')
    assert chain.residues[0].name == 'W'
