"""
Boltz-2 folding oracle: a nucleic-acid-capable FoldingOracle, parallel to ESMFold.

Unlike ESMFold (protein only), Boltz-2 co-folds protein / DNA / RNA / ligand
complexes. This oracle mirrors the ESMFold pair (ESMFold / ESMFoldResult) so the
swap is transparent to the energy layer: BoltzResult declares the same five
fields every FoldingOracle energy term requires (input_chains, structure,
local_plddt, ptm, pae), plus an optional `iptm` (interface pTM) that has no
ESMFold equivalent.

The molecule type of each chain is carried to boileroom via the `options` dict
("molecule_types"), which the (patched) Boltz wrapper emits as the FASTA entity
type per chain. See the companion boileroom `_sequences_to_fasta` change.

NOTE on boileroom version: BAGEL pins boileroom==0.2.2, which predates Boltz-2.
Using this oracle requires bumping boileroom to a version that ships
`boileroom.models.boltz`. In that newer API the wrapper constructor takes
`backend` first (`Boltz2(backend=..., config=...)`); the single backend call is
isolated in `_remote_fold`/`_load` so there is exactly one place to reconcile
against the target boileroom version.

MIT License
"""

import pathlib as pl
import numpy as np
import numpy.typing as npt
from typing import List, Any, Type, Optional
from pydantic import field_validator

from ...chain import Chain
from .utils import reindex_chains, validate_array_range
from .base import FoldingOracle, FoldingResult

from boileroom import app  # type: ignore
from boileroom.models.boltz.boltz2 import Boltz2 as Boltz2Boiler  # type: ignore
from boileroom.models.boltz.types import Boltz2Output  # type: ignore
from modal import App

from biotite.structure import AtomArray
import logging

logger = logging.getLogger(__name__)


class BoltzResult(FoldingResult):
    """
    Stores statistics from the Boltz-2 folding algorithm.

    Same contract as ESMFoldResult, so every FoldingOracle energy term accepts it
    unchanged. `iptm` is an optional extra (interface pTM) for binding design.
    """

    input_chains: list[Chain]
    structure: AtomArray  # structure of the predicted complex (may contain protein + DNA/RNA)
    local_plddt: npt.NDArray[np.float64]  # per-residue predicted LDDT (0 to 1), shape [1, n_res]
    ptm: npt.NDArray[np.float64]  # (global) predicted TM score (0 to 1), shape [1]
    pae: npt.NDArray[np.float64]  # pairwise predicted alignment error (Angstroms), shape [1, n_res, n_res]
    iptm: Optional[npt.NDArray[np.float64]] = None  # interface pTM (0 to 1); no ESMFold equivalent

    @field_validator('local_plddt')
    def validate_local_plddt(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'local_plddt', 0, 1)

    @field_validator('ptm')
    def validate_ptm(cls, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return validate_array_range(v, 'ptm', 0, 1)

    def save_attributes(self, filepath: pl.Path) -> None:
        np.savetxt(filepath.with_suffix('.plddt'), self.local_plddt[0], fmt='%.6f', header='plddt')
        np.savetxt(filepath.with_suffix('.pae'), self.pae[0], fmt='%.6f', header='pae')


class BoltzOracle(FoldingOracle):
    """
    Object that uses Boltz-2 to predict structures of protein / DNA / RNA complexes.
    """

    result_class: Type[BoltzResult] = BoltzResult

    def __init__(self, use_modal: bool = False, config: dict[str, Any] = {}, modal_app_context: App | None = None):
        self.use_modal = use_modal
        self.modal_app_context = modal_app_context
        # Boltz's boileroom wrapper is single-sample (no multi-sample support yet),
        # which already satisfies BAGEL's batch==1 contract on local_plddt/pae.
        self.default_config: dict[str, Any] = {}
        self._load(config)

        if self.use_modal and self.modal_app_context is None:
            import atexit

            atexit.register(self.__del__)

    def __del__(self) -> None:
        if self.use_modal and hasattr(self, 'modal_app_context') and self.modal_app_context is not None:  # type: ignore
            self.modal_app_context.__exit__(None, None, None)  # type: ignore
            self.modal_app_context = None

    def _load(self, config: dict[str, Any] = {}) -> None:
        if self.use_modal and self.modal_app_context is None:
            self.modal_app_context = app.run()
            self.modal_app_context.__enter__()  # type: ignore
        config = {**self.default_config, **config}
        # Version-dependent: newer boileroom takes `backend` first; pass config by keyword.
        self.model = Boltz2Boiler(config=config)

    def _pre_process(self, chains: list[Chain]) -> list[str]:
        """Colon-joined multimer string, identical convention to ESMFold."""
        monomers = [chain.sequence for chain in chains]
        return [':'.join(monomers)]

    def _molecule_types(self, chains: list[Chain]) -> list[str]:
        """Per-chain molecule types, in the same order as the colon-joined sequence."""
        return [chain.molecule_type for chain in chains]

    def fold(self, chains: List[Chain]) -> BoltzResult:
        sequence = self._pre_process(chains)
        options = {'molecule_types': self._molecule_types(chains)}
        return self._reduce_output(self._remote_fold(sequence, options), chains)

    def _remote_fold(self, sequence: List[str], options: dict[str, Any]) -> Boltz2Output:
        # The single boileroom-version-dependent call site. Newer boileroom dispatches
        # backend inside `.fold`; 0.2.x-style modal Functions expose `.fold.remote`.
        return self.model.fold(sequence, options=options)

    def _reduce_output(self, output: Boltz2Output, chains: List[Chain]) -> BoltzResult:
        """
        Reduce a Boltz2Output to a BoltzResult.

        Differs from ESMFold's reducer in two ways (Step 2/4 findings):
          * plddt is already PER-RESIDUE and unit-scaled, so there is NO CA-atom
            indexing; we use it directly and add the batch axis.
          * Boltz returns Python lists over samples, so we take sample 0 and wrap
            to the batched shapes BAGEL's energy terms assert (leading axis == 1).
        """
        atoms = reindex_chains(output.atom_array, [chain.chain_ID for chain in chains])

        plddt = np.asarray(output.plddt[0], dtype=np.float64)  # (n_res,), unit-scaled by boileroom
        pae = np.asarray(output.pae[0], dtype=np.float64)  # (n_res, n_res)
        ptm = np.asarray(output.ptm[0], dtype=np.float64)  # (1,) already

        iptm: Optional[npt.NDArray[np.float64]] = None
        if getattr(output, 'iptm', None) is not None and output.iptm[0] is not None:
            iptm = np.asarray(output.iptm[0], dtype=np.float64)  # (1,)

        return self.result_class(
            input_chains=chains,
            structure=atoms,
            local_plddt=plddt[None, :],  # (1, n_res) -- NO atom_order['CA'] indexing
            ptm=ptm,  # (1,)
            pae=pae[None, ...],  # (1, n_res, n_res)
            iptm=iptm,
        )
