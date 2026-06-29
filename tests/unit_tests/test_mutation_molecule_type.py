"""Tests for type-aware mutation (DNA/RNA bias routing).

The substitution path needs only a Chain; the addition/removal paths take a
`system` but only touch `system.states`, so a stand-in with no states exercises
the sampling while skipping energy-term bookkeeping. GPU-free; run with
`pytest --oracles skip`.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import bagel as bg
from bagel import constants as C
from bagel.mutation import Canonical, GrandCanonical


def protein_chain() -> bg.Chain:
    return bg.Chain([bg.Residue(a, 'P', i) for i, a in enumerate('ACDEFGHIK')])


def dna_chain() -> bg.Chain:
    return bg.Chain([bg.Residue(b, 'D', i, molecule_type='dna') for i, b in enumerate('ACGTACGT')])


def rna_chain() -> bg.Chain:
    return bg.Chain([bg.Residue(b, 'R', i, molecule_type='rna') for i, b in enumerate('ACGUACGU')])


def test_bias_routing_and_protein_unchanged() -> None:
    can = Canonical()
    assert can._bias_for('protein') is can.mutation_bias
    assert C.MUTATION_BIASES['protein'] is C.mutation_bias_no_cystein
    assert set(can._bias_for('dna').keys()) == set('ACGT')
    assert set(can._bias_for('rna').keys()) == set('ACGU')
    assert abs(sum(can._bias_for('dna').values()) - 1.0) < 1e-9


def test_unknown_molecule_type_bias_rejected() -> None:
    with pytest.raises(AssertionError):
        Canonical()._bias_for('peptide')


@pytest.mark.parametrize('make_chain,alphabet', [(dna_chain, 'ACGT'), (rna_chain, 'ACGU')])
def test_substitution_stays_in_alphabet(make_chain, alphabet: str) -> None:
    can = Canonical()
    chain = make_chain()
    np.random.seed(0)
    for _ in range(200):
        mut = can.mutate_random_residue(chain)
        assert mut.new_amino_acid in alphabet
    assert set(chain.sequence) <= set(alphabet)


def test_protein_substitution_behaviour_unchanged() -> None:
    can = Canonical()
    chain = protein_chain()
    np.random.seed(0)
    for _ in range(200):
        mut = can.mutate_random_residue(chain)
        assert mut.new_amino_acid in C.aa_dict
        assert mut.new_amino_acid != mut.old_amino_acid  # exclude_self
        assert mut.new_amino_acid != 'C'  # no-cysteine default


def test_grandcanonical_add_and_remove_respect_type() -> None:
    gc = GrandCanonical()
    fake_system = SimpleNamespace(states=[])
    chain = dna_chain()
    start = chain.length
    for _ in range(50):
        gc.add_random_residue(chain, fake_system)
    assert chain.length == start + 50
    assert set(chain.sequence) <= set('ACGT')
    for _ in range(20):
        gc.remove_random_residue(chain, fake_system)
    assert chain.molecule_type == 'dna'
    assert set(chain.sequence) <= set('ACGT')


def test_mixed_system_each_chain_stays_in_its_alphabet() -> None:
    can = Canonical()
    p, d = protein_chain(), dna_chain()
    np.random.seed(1)
    for _ in range(100):
        can.mutate_random_residue(p)
        can.mutate_random_residue(d)
    assert set(p.sequence) <= set(C.aa_dict)
    assert set(d.sequence) <= set('ACGT')
