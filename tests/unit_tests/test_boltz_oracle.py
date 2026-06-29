"""Tests for the Boltz-2 folding oracle (contract + output reshaping).

These tests are GPU-free: they never call the model. They verify the two things
that must hold for the oracle swap to be transparent to the energy layer:
  1. BoltzResult declares every field the folding energy terms require.
  2. _reduce_output reshapes Boltz's per-sample lists into the batched arrays
     BAGEL asserts, with NO CA-atom indexing (plddt is already per-residue).

Requires a boileroom version that ships `boileroom.models.boltz`.
Run with `pytest --oracles skip`.
"""

import numpy as np
import pytest
import biotite.structure as struc

import bagel as bg
from bagel.oracles.folding.boltz import BoltzOracle, BoltzResult


REQUIRED_FOLDING_FIELDS = {'input_chains', 'structure', 'local_plddt', 'ptm', 'pae'}


class _FakeBoltzOutput:
    """Mimics Boltz2Output: per-sample lists, plddt already per-residue & unit-scaled."""

    def __init__(self) -> None:
        self.atom_array = [
            struc.array(
                [
                    struc.Atom([0.0, 0, 0], chain_id='X', res_id=1, res_name='ALA', atom_name='CA', element='C'),
                    struc.Atom([1.0, 0, 0], chain_id='Y', res_id=1, res_name='DG', atom_name="C1'", element='C'),
                ]
            )
        ]
        self.plddt = [np.array([0.5, 0.6, 0.7], dtype=np.float32)]
        self.pae = [np.arange(9, dtype=np.float32).reshape(3, 3)]
        self.ptm = [np.array([0.82], dtype=np.float32)]
        self.iptm = [np.array([0.71], dtype=np.float32)]


def _chains() -> list[bg.Chain]:
    prot = bg.Chain([bg.Residue('A', 'A', 0)])
    dna = bg.Chain([bg.Residue('G', 'B', 0, molecule_type='dna')])
    return [prot, dna]


def test_boltz_result_satisfies_folding_contract() -> None:
    fields = set(BoltzResult.model_fields.keys())
    assert REQUIRED_FOLDING_FIELDS <= fields
    assert 'iptm' in fields  # bonus interface metric, optional


def test_reduce_output_reshapes_to_batched_arrays() -> None:
    orc = BoltzOracle(use_modal=False)
    res = orc._reduce_output(_FakeBoltzOutput(), _chains())
    assert res.local_plddt.shape == (1, 3)
    assert res.pae.shape == (1, 3, 3)
    assert res.ptm.shape == (1,)
    assert res.iptm is not None and res.iptm.shape == (1,)


def test_reduce_output_uses_plddt_directly_no_ca_indexing() -> None:
    orc = BoltzOracle(use_modal=False)
    res = orc._reduce_output(_FakeBoltzOutput(), _chains())
    assert np.allclose(res.local_plddt[0], [0.5, 0.6, 0.7])


def test_structure_reindexed_and_keeps_nucleotides() -> None:
    orc = BoltzOracle(use_modal=False)
    res = orc._reduce_output(_FakeBoltzOutput(), _chains())
    assert set(np.unique(res.structure.chain_id)) == {'A', 'B'}
    assert set(map(str, np.unique(res.structure.res_name))) == {'ALA', 'DG'}


def test_plddt_out_of_unit_range_is_rejected() -> None:
    orc = BoltzOracle(use_modal=False)
    out = _FakeBoltzOutput()
    out.plddt = [np.array([55.0, 60.0, 70.0])]  # 0-100 scale would be a loud failure
    with pytest.raises(Exception, match='between 0'):
        orc._reduce_output(out, _chains())
