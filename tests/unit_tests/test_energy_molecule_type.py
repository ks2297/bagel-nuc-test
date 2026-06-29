"""Tests for the energy-term triage (DNA/RNA support).

Two mechanisms:
  * the backbone-atom union lets geometric (Group-B) terms work on nucleic acids
    instead of masking to an empty set (NaN);
  * protein-chemistry (Group-C) terms declare supported_molecule_types={'protein'}
    and reject nucleic-acid targets loudly at construction.

GPU-free (no model call). Run with `pytest --oracles skip`.
"""

import numpy as np
import pytest
import biotite.structure as struc

import bagel.energies as E
from bagel import constants as C
from bagel.chain import Residue
from bagel.oracles.folding.base import FoldingOracle, FoldingResult


class _FakeFoldingOracle(FoldingOracle):
    """Minimal FoldingOracle: result_class has 'structure', satisfies isinstance checks."""

    result_class = FoldingResult

    def fold(self, chains):  # pragma: no cover - never called
        raise NotImplementedError


@pytest.fixture
def oracle() -> _FakeFoldingOracle:
    return _FakeFoldingOracle()


def _dna_residues() -> list[Residue]:
    return [Residue(b, 'B', i, molecule_type='dna') for i, b in enumerate('ACGT')]


def _protein_residues() -> list[Residue]:
    return [Residue(a, 'A', i) for i, a in enumerate('ACDE')]


# --------------------------------------------------------------------------- #
# Backbone-atom generalisation (Group B)
# --------------------------------------------------------------------------- #
def test_union_contains_both_backbones() -> None:
    assert set(('CA', 'N', 'C')) <= set(C.all_backbone_atoms)
    assert 'P' in C.all_backbone_atoms and "C4'" in C.all_backbone_atoms


def test_protein_backbone_mask_unchanged_under_union() -> None:
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


def test_dna_backbone_centroid_is_finite() -> None:
    dna = struc.array(
        [
            struc.Atom([float(i), 0, 0], chain_id='B', res_id=1, res_name='DG', atom_name=n, element='C')
            for i, n in enumerate(('P', "O5'", "C5'", "C4'", "C3'", "O3'", 'N9'))
        ]
    )
    mask = np.isin(dna.atom_name, C.all_backbone_atoms)
    assert mask.sum() == 6  # the six phosphodiester backbone atoms
    assert not np.any(np.isnan(np.mean(dna[mask].coord, axis=0)))
    # the old protein-only set would have masked to empty -> NaN
    assert not np.isin(dna.atom_name, ('CA', 'N', 'C')).any()


# --------------------------------------------------------------------------- #
# Group-C applicability gate
# --------------------------------------------------------------------------- #
def test_group_c_rejects_nucleic_residues(oracle) -> None:
    dna = _dna_residues()
    with pytest.raises(AssertionError, match='supports molecule types'):
        E.HydrophobicEnergy(oracle, residues=dna)
    with pytest.raises(AssertionError, match='supports molecule types'):
        E.HydropathyEnergy(oracle, residues=dna)
    with pytest.raises(AssertionError, match='supports molecule types'):
        E.SecondaryStructureEnergy(oracle, residues=dna, target_secondary_structure='alpha-helix')


def test_group_c_accepts_protein_residues(oracle) -> None:
    prot = _protein_residues()
    E.HydrophobicEnergy(oracle, residues=prot)
    E.HydropathyEnergy(oracle, residues=prot)
    E.SecondaryStructureEnergy(oracle, residues=prot, target_secondary_structure='alpha-helix')


def test_group_b_term_accepts_nucleic_residues(oracle) -> None:
    # geometric terms carry the permissive default and now compute correct NA backbones
    dna = _dna_residues()
    E.SeparationEnergy(oracle, residues=(dna, dna))


def test_whole_system_group_c_still_constructs(oracle) -> None:
    # residues=None cannot be gated at construction; self-neutralises at compute time
    E.HydrophobicEnergy(oracle)
