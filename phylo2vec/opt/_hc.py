"""Methods for hill-climbing optimisation."""
import os

from pathlib import Path

import numpy as np

from joblib import delayed, effective_n_jobs, Parallel

from phylo2vec.opt._base import BaseOptimizer
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
    reorder_method : str, optional
        Phylo2Vec vector reordering method, by default "birth_death"
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
        reorder_method="birth_death",
        random_seed=None,
        tol=0.001,
        patience=3,
        rounds=1,
        n_jobs=None,
        verbose=False,
    ):
        super().__init__(random_seed=random_seed)

        if tree_folder_path is None:
            os.makedirs("trees", exist_ok=True)
            tree_folder_path = "trees"

        self.raxml_cmd = raxml_cmd
        self.tree_folder_path = tree_folder_path
        self.substitution_model = substitution_model
        self.reorder_method = reorder_method
        self.tol = tol
        self.rounds = rounds
        self.patience = patience
        self.n_jobs = effective_n_jobs(n_jobs)
        self.verbose = verbose

    def _optimise(self, fasta_path, v, taxa_dict):
        current_loss = raxml_loss(
            v=v,
            taxa_dict=taxa_dict,
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

            v_proposal, proposal_loss, taxa_dict = self._optimise_single(
                fasta_path, v.copy(), taxa_dict
            )

            v = v_proposal.copy()

            if proposal_loss - current_loss < self.tol:
                # Found a better loss so reset the patience counter
                current_loss = proposal_loss

                wait = 0
            else:
                # No drastic increase so increment the patience counter
                wait += 1

                if self.verbose:
                    print(f"No significantly better loss found {wait}/{self.patience}.")

            losses.append(current_loss)

        return v, taxa_dict, losses

    def _optimise_single(self, fasta_path, v, taxa_dict):
        # Reorder v
        v_shuffled, taxa_dict = reorder_v(self.reorder_method, v, taxa_dict)

        # Get current loss
        current_best_loss = raxml_loss(
            v=v_shuffled,
            taxa_dict=taxa_dict,
            fasta_path=fasta_path,
            tree_folder_path=self.tree_folder_path,
            substitution_model=self.substitution_model,
            outfile=f"{Path(fasta_path).stem}_best.tree",
            cmd=self.raxml_cmd,
        )

        if self.verbose:
            print(f"Start optimise_single: {current_best_loss:.3f}")

        for _ in range(self.rounds):
            for i in reversed(range(1, len(v_shuffled))):
                # Calculate gradient for changes in row i
                # "gradient" here simply refers to a numerical gradient
                # between loss(v_current) and loss(v_proposal)
                proposal_grads, proposal_losses = self.grad_single(
                    fasta_path=fasta_path,
                    v_proposal=v_shuffled,
                    current_loss=current_best_loss,
                    taxa_dict=taxa_dict,
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

        return v_shuffled, current_best_loss, taxa_dict

    def grad_single(self, fasta_path, v_proposal, current_loss, taxa_dict, i):
        """Calculate gradients for a single index of v

        Parameters
        ----------
        fasta_path : str
            Path to fasta file
        v_proposal : numpy.ndarray or list
            v representation of a new tree proposal
        current_loss : float
            Current best loss
        taxa_dict : dict[int, str]
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
                taxa_dict=taxa_dict,
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
