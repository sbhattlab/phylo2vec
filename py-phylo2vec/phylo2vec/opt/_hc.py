"""Methods for hill-climbing optimisation."""

import os

from pathlib import Path

import numpy as np

from joblib import delayed, Parallel

from phylo2vec.opt._base import BaseOptimizer, BaseResult
from phylo2vec.opt._hc_losses import raxml_loss
from phylo2vec.utils.vector import reorder_v, reroot_at_random


class HillClimbingOptimizer(BaseOptimizer):
    """Optimisation using a simple hill-climbing scheme

    More details in the Phylo2Vec paper/preprint

    Parameters
    ----------
    raxml_cmd : str
        Location of the RAxML-nG executable, by default "raxml-ng"
    tree_folder_path : str, optional
        Path to a folder which will contain all intermediary and best trees, by default None
        If None, will create a folder called "trees"
    substitution_model : str, optional
        DNA/AA substitution model, by default "GTR"
    random_seed : int, optional
        Random seed, by default None
        Controls both the randomness of the initial vector
    tol : float, optional
        Tolerance for topology change, by default 0.001
    patience : int, optional
        Number of passes without improvement, by default 3
    rounds : int, optional
        Number of indices of v to change in a single pass, by default 1
    n_jobs : int, optional
        The number of jobs to run in parallel, by default None
    verbose : bool, optional
        Controls the verbosity when optimising, by default False
    """

    def __init__(
        self,
        raxml_cmd="raxml-ng",
        tree_folder_path=None,
        substitution_model="GTR",
        random_seed=None,
        tol=0.001,
        patience=3,
        rounds=1,
        n_jobs=None,
        verbose=False,
    ):
        super().__init__(random_seed=random_seed, n_jobs=n_jobs, verbose=verbose)

        if tree_folder_path is None:
            os.makedirs("trees", exist_ok=True)
            tree_folder_path = "trees"

        self.raxml_cmd = raxml_cmd
        self.tree_folder_path = tree_folder_path
        self.substitution_model = substitution_model
        self.tol = tol
        self.rounds = rounds
        self.patience = patience
        self.verbose = verbose

    def _optimise(self, fasta_path, v, label_mapping):
        current_loss = raxml_loss(
            v=v,
            label_mapping=label_mapping,
            fasta_path=fasta_path,
            tree_folder_path=self.tree_folder_path,
            substitution_model=self.substitution_model,
            cmd=self.raxml_cmd,
        )

        wait = 0

        losses = [current_loss]

        while wait < self.patience:
            if self.verbose:
                print("Changing equivalences...")
            v_proposal = reroot_at_random(v)

            v_proposal, proposal_loss, label_mapping = self._step(
                fasta_path, v.copy(), label_mapping
            )

            v = v_proposal.copy()

            if abs(proposal_loss - current_loss) > self.tol:
                # Found a better loss so reset the patience counter
                current_loss = proposal_loss

                wait = 0
            else:
                # No drastic increase so increment the patience counter
                wait += 1

                if self.verbose:
                    print(f"No significantly better loss found {wait}/{self.patience}.")

            losses.append(current_loss)

        best_params = BaseResult(
            v=v,
            label_mapping=label_mapping,
            scores=losses,
            best_score=current_loss,
        )

        return best_params

    def _step(self, fasta_path, v, label_mapping):
        # Reorder v
        v_shuffled, label_mapping = reorder_v("birth_death", v, label_mapping)

        # Get current loss
        current_best_loss = raxml_loss(
            v=v_shuffled,
            label_mapping=label_mapping,
            fasta_path=fasta_path,
            tree_folder_path=self.tree_folder_path,
            substitution_model=self.substitution_model,
            outfile=f"{Path(fasta_path).stem}_best.tree",
            cmd=self.raxml_cmd,
        )

        if self.verbose:
            print(f"Start optimise_single: {current_best_loss:.3f}")

        for _ in range(self.rounds):
            for i in range(1, len(v_shuffled)):
                # Calculate gradient for changes in row i
                # "gradient" here simply refers to a numerical gradient
                # between loss(v_current) and loss(v_proposal)
                proposal_grads, proposal_losses = self._value_and_grad_proposals(
                    fasta_path=fasta_path,
                    v_proposal=v_shuffled,
                    current_loss=current_best_loss,
                    label_mapping=label_mapping,
                    i=i,
                )

                # find index of max gradient
                grad_choice = proposal_grads.argmax(0)

                # Is there a positive gradient?
                if proposal_grads[grad_choice] > 0:
                    # Discrete gradient step
                    v_shuffled[i] = (
                        grad_choice + 1 if grad_choice >= v_shuffled[i] else grad_choice
                    )

                    # Reorder v
                    # v_shuffled, taxa_dict = reorder_v(v_shuffled, taxa_dict)

                    if self.verbose:
                        grad_propose = proposal_losses[grad_choice] - current_best_loss
                        print(
                            f"Loss: {proposal_losses[grad_choice]:.3f} (diff: {grad_propose:.3f})"
                        )

                    # Update best loss
                    current_best_loss = proposal_losses[grad_choice]

        if self.verbose:
            print(f"End optimise_single: {current_best_loss:.3f}")

        return v_shuffled, current_best_loss, label_mapping

    def _value_and_grad_proposals(
        self, fasta_path, v_proposal, current_loss, label_mapping, i
    ):
        """Calculate losses and gradients for a single index change in v

        Parameters
        ----------
        fasta_path : str
            Path to fasta file
        v_proposal : numpy.ndarray or list
            v representation of a new tree proposal
        current_loss : float
            Current best loss
        label_mapping : Dict[int, str]
            Current mapping of leaf labels (integer) to taxa
        i : long
            index of v to change and to calculate a gradient on

        NOTE: # "gradient" here simply refers to a numerical gradient
                between loss(v_current) and loss(v_proposal)

        Returns
        -------
        proposa_losses_diff : numpy.ndarray
            Difference between the current loss and the proposal losses
        proposal_losses : numpy.ndarray
            Proposal losses
        """

        v_copy = v_proposal.copy()

        def run(v_other, i, j):
            v_other[i] = j
            return raxml_loss(
                v=v_other,
                label_mapping=label_mapping,
                fasta_path=fasta_path,
                tree_folder_path=self.tree_folder_path,
                substitution_model=self.substitution_model,
                outfile=f"{Path(fasta_path).stem}_tree{i}{j}.tree",
                cmd=self.raxml_cmd,
            )

        proposal_losses = np.array(
            Parallel(n_jobs=self.n_jobs)(
                delayed(run)(v_copy, i, j)
                for j in range(2 * i + 1)
                if j != v_proposal[i]
            )
        )

        proposal_losses_diff = current_loss - proposal_losses

        return proposal_losses_diff, proposal_losses
