"""
Step 11 — end-to-end run: design a protein that binds a fixed DNA target.

This exercises the more interesting case: a MIXED system (protein + DNA) folded
together as a complex by Boltz, where only the protein is mutated and the DNA is
held fixed.

  representation (protein chain + DNA chain) -> type-aware mutation (only the
  protein is mutable) -> BoltzOracle co-folds the protein-DNA complex
  -> binding + confidence energy -> Monte-Carlo optimisation.

Objective: minimise the binder<->target distance (FlexEvoBind, the EvoBind-style
loss, pLDDT-weighted) while keeping the fold confident (overall pLDDT).

REQUIREMENTS: a GPU + the modified boileroom that ships Boltz-2 (see README).
This will NOT run without a GPU.

Run:  python examples/protein_dna_binder.py
"""

import bagel as bg

# ----- EDIT THESE FOR YOUR ENVIRONMENT --------------------------------------
USE_MODAL = True
N_STEPS = 20
START_BINDER = 'MKQLEDKVEELLSKNYHLENEVARLKKLVGER'   # 32 aa starting binder
DNA_TARGET = 'GCGCAATTGCGC'                          # fixed 12 bp DNA target
# ----------------------------------------------------------------------------


def main() -> None:
    # Protein binder: mutable.
    binder_residues = [
        bg.Residue(name=aa, chain_ID='P', index=i, mutable=True, molecule_type='protein')
        for i, aa in enumerate(START_BINDER)
    ]
    binder = bg.Chain(binder_residues)

    # DNA target: NOT mutable (mutable=False) so the optimiser never changes it.
    target_residues = [
        bg.Residue(name=base, chain_ID='D', index=i, mutable=False, molecule_type='dna')
        for i, base in enumerate(DNA_TARGET)
    ]
    target = bg.Chain(target_residues)

    oracle = bg.oracles.folding.BoltzOracle(use_modal=USE_MODAL)

    state = bg.State(
        name='complex',
        chains=[binder, target],
        energy_terms=[
            # pull the binder onto the DNA target (minimise inter-group distance)
            bg.energies.FlexEvoBindEnergy(
                oracle=oracle,
                residues=(binder_residues, target_residues),
                plddt_weighted=True,
                weight=1.0,
            ),
            # keep the complex confidently folded
            bg.energies.OverallPLDDTEnergy(oracle=oracle, weight=0.5),
        ],
    )
    system = bg.System(states=[state], name='protein_dna_binder')

    # Canonical substitutions. choose_chain weights by MUTABLE residues, and the DNA
    # target has none, so only the protein binder is ever mutated (from the protein
    # alphabet, because that chain is molecule_type='protein').
    minimizer = bg.minimizer.SimulatedAnnealing(
        mutator=bg.mutation.Canonical(),
        initial_temperature=1.0,
        final_temperature=0.01,
        n_steps=N_STEPS,
        log_path='protein_dna_binder_run',
    )

    print(f'Target DNA (fixed): {target.sequence}')
    print(f'Starting binder:    {binder.sequence}')
    best = minimizer.minimize_system(system)
    print(f'Optimised binder:   {best.states[0].chains[0].sequence}')
    print(f'Target unchanged:   {best.states[0].chains[1].sequence}')


if __name__ == '__main__':
    main()
