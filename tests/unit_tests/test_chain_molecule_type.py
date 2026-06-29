"""Tests for the molecule_type representation layer (DNA/RNA support).

These tests require no oracle, so they run with `pytest --oracles skip`.
They cover two things: that the protein path is unchanged, and that the
new DNA/RNA path validates, maps three-letter names, and stays homogeneous.
"""

import pytest
import bagel as bg
from bagel import constants as C


# --------------------------------------------------------------------------- #
# Protein path is unchanged (the Step-3 guarantee, now enforced)
# --------------------------------------------------------------------------- #
def test_protein_alphabet_is_the_same_object() -> None:
    # The 'protein' entry IS aa_dict, so protein validation/mapping is byte-identical.
    assert C.ALPHABETS['protein'] is C.aa_dict


def test_residue_defaults_to_protein() -> None:
    r = bg.Residue('A', 'X', 0)
    assert r.molecule_type == 'protein'
    assert r.three_letter_name == 'ALA'


def test_protein_chain_unchanged() -> None:
    chain = bg.Chain([bg.Residue(aa, 'X', i) for i, aa in enumerate('ACDEFG')])
    assert chain.molecule_type == 'protein'
    assert chain.sequence == 'ACDEFG'


# --------------------------------------------------------------------------- #
# DNA / RNA residues
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    'mol_type,name,three',
    [('dna', 'A', 'DA'), ('dna', 'G', 'DG'), ('dna', 'T', 'DT'), ('rna', 'U', 'U'), ('rna', 'G', 'G')],
)
def test_nucleotide_three_letter_names(mol_type: str, name: str, three: str) -> None:
    assert bg.Residue(name, 'X', 0, molecule_type=mol_type).three_letter_name == three


def test_collision_letter_resolves_by_type() -> None:
    # 'G' is glycine in a protein chain but guanine in a DNA chain - disambiguated by the tag.
    assert bg.Residue('G', 'X', 0, molecule_type='protein').three_letter_name == 'GLY'
    assert bg.Residue('G', 'X', 0, molecule_type='dna').three_letter_name == 'DG'


def test_rna_U_accepted_but_rejected_in_protein_and_dna() -> None:
    assert bg.Residue('U', 'X', 0, molecule_type='rna').three_letter_name == 'U'
    with pytest.raises(AssertionError):
        bg.Residue('U', 'X', 0, molecule_type='protein')
    with pytest.raises(AssertionError):
        bg.Residue('U', 'X', 0, molecule_type='dna')


def test_unknown_molecule_type_rejected() -> None:
    with pytest.raises(AssertionError):
        bg.Residue('A', 'X', 0, molecule_type='peptide')


# --------------------------------------------------------------------------- #
# DNA / RNA chains
# --------------------------------------------------------------------------- #
def test_dna_chain_properties_and_mutation() -> None:
    chain = bg.Chain([bg.Residue(b, 'D', i, molecule_type='dna') for i, b in enumerate('ACGT')])
    assert chain.molecule_type == 'dna'
    assert chain.sequence == 'ACGT'
    chain.mutate_residue(0, 'T')
    assert chain.residues[0].three_letter_name == 'DT'
    chain.add_residue('C', -1)
    assert chain.residues[-1].molecule_type == 'dna'


@pytest.mark.parametrize('bad', ['U', 'E', 'K'])
def test_dna_chain_rejects_non_dna_letters(bad: str) -> None:
    chain = bg.Chain([bg.Residue(b, 'D', i, molecule_type='dna') for i, b in enumerate('ACGT')])
    with pytest.raises(AssertionError):
        chain.mutate_residue(1, bad)


def test_chain_molecule_type_must_be_homogeneous() -> None:
    with pytest.raises(AssertionError):
        bg.Chain(
            [
                bg.Residue('A', 'M', 0, molecule_type='protein'),
                bg.Residue('A', 'M', 1, molecule_type='dna'),
            ]
        )
