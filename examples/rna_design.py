"""
Step 11 — minimal end-to-end run: design an RNA that folds confidently.

This is the smallest possible "real run" that exercises the whole DNA/RNA path:
  representation (an RNA chain) -> type-aware mutation -> BoltzOracle folding RNA
  -> confidence energy terms -> Monte-Carlo optimisation.

The objective is "find an RNA sequence Boltz folds with high confidence": we add
PTM + overall-pLDDT energy terms (both are negated internally, so MINIMISING the
energy MAXIMISES confidence).

REQUIREMENTS: a GPU + the modified boileroom that ships Boltz-2 (see README).
This will NOT run without a GPU.

Run:  python examples/rna_design.py
"""

import bagel as bg

# ----- EDIT THESE FOR YOUR ENVIRONMENT --------------------------------------
USE_MODAL = True       # how your group runs boileroom's Boltz backend
N_STEPS = 20           # keep small for a first smoke test; raise once it works
START_RNA = 'ACGUACGUACGUACGU'   # 16 nt starting sequence (will be optimised)
# ----------------------------------------------------------------------------


def main() -> None:
    # One RNA chain, all residues mutable, tagged molecule_type='rna'.
    residues = [
        bg.Residue(name=base, chain_ID='R', index=i, mutable=True, molecule_type='rna')
        for i, base in enumerate(START_RNA)
    ]
    rna_chain = bg.Chain(residues)

    # Nucleic-acid-capable oracle (the Step-8 oracle).
    oracle = bg.oracles.folding.BoltzOracle(use_modal=USE_MODAL)

    # Confidence objective (both Group-A terms; type-agnostic, work on RNA).
    state = bg.State(
        name='rna_fold',
        chains=[rna_chain],
        energy_terms=[
            bg.energies.OverallPLDDTEnergy(oracle=oracle, weight=1.0),
            bg.energies.PTMEnergy(oracle=oracle, weight=1.0),
        ],
    )
    system = bg.System(states=[state], name='rna_design')

    # Canonical = substitutions only (fixed length). Mutation is type-aware, so it
    # draws from the RNA alphabet automatically because the chain is molecule_type='rna'.
    minimizer = bg.minimizer.SimulatedAnnealing(
        mutator=bg.mutation.Canonical(),
        initial_temperature=1.0,
        final_temperature=0.01,
        n_steps=N_STEPS,
        log_path='rna_design_run',
    )

    print(f'Starting RNA: {rna_chain.sequence}')
    best = minimizer.minimize_system(system)
    print(f'Optimised RNA: {best.states[0].chains[0].sequence}')


if __name__ == '__main__':
    main()
