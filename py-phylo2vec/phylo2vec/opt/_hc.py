"""Methods for hill-climbing optimisation."""

import os

from pathlib import Path

import numpy as np

from joblib import delayed, Parallel

from phylo2vec.opt._api import register_method
from phylo2vec.opt._base import BaseOptimizer, BaseResult
from phylo2vec.opt._hc_losses import raxml_loss
from phylo2vec.utils.vector import reorder_v, reroot_at_random


@register_method()
class HillClimbing(BaseOptimizer):
    """Optimisation using a simple hill-climbing scheme

    More details in the Phylo2Vec paper/preprint

    Parameters
    ----------
    model : str, optional
        DNA/AA substitution model, by default "GTR"
    tol : float, optional
        Tolerance for topology change, by default 0.001
    patience : int, optional
        Number of passes without improvement, by default 3
    rounds : int, optional
        Number of indices of v to change in a single pass, by default 1
    tree_folder_path : str, optional
        Path to a folder which will contain all intermediary and best trees, by default None
        If None, will create a folder called "trees"
    random_seed : int, optional
        Random seed, by default None
        Controls both the randomness of the initial vector
    n_jobs : int, optional
        The number of jobs to run in parallel, by default None
    verbose : bool, optional
        Controls the verbosity when optimising, by default False
    """

    def __init__(
        self,
        model,
        tol=0.001,
        rounds=1,
        patience=3,
        tree_folder_path=None,
        random_seed=None,
        n_jobs=None,
        verbose=False,
        **loss_kwargs,
    ):
        super().__init__(
            mode="vector", random_seed=random_seed, n_jobs=n_jobs, verbose=verbose
        )

        if tree_folder_path is None:
            os.makedirs("trees", exist_ok=True)
            tree_folder_path = "trees"

        self.model = model
        self.rounds = rounds
        self.patience = patience
        self.tol = tol
        self.tree_folder_path = tree_folder_path
        self.verbose = verbose
        self.loss_kwargs = loss_kwargs

    def _optimise(self, fasta_path, tree, label_mapping) -> BaseResult:
        """Optimise a tree using hill-climbing

        Parameters
        ----------
        fasta_path : str
            Path to fasta file
        tree : numpy.ndarray
            random tree to optimize, in v representation
        label_mapping : Dict[int, str]
            Current mapping of leaf labels (integer) to taxa

        Returns
        -------
        BaseResult
            best : numpy.ndarray
                Optimized phylo2vec vector.
            label_mapping : Dict[int, str]
                Mapping of leaf labels (integer) to taxa.
            best_score : float
                The best score achieved during optimization.
            scores : List[float]
                List of scores obtained during optimization.
        """
        current_loss = raxml_loss(
            v=tree,
            label_mapping=label_mapping,
            fasta_path=fasta_path,
            model=self.model,
            tree_folder_path=self.tree_folder_path,
            **self.loss_kwargs,
        )

        wait = 0

        losses = [current_loss]

        while wait < self.patience:
            if self.verbose:
                print("Changing equivalences...")
            v_proposal = reroot_at_random(tree)

            v_proposal, proposal_loss, label_mapping = self._step(
                fasta_path, tree.copy(), label_mapping
            )

            tree = v_proposal.copy()

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

        return BaseResult(
            best=tree,
            label_mapping=label_mapping,
            scores=losses,
            best_score=current_loss,
        )

    def _step(self, fasta_path, v, label_mapping):
        """Perform a step of the hill-climbing optimisation

        Parameters
        ----------
        fasta_path : str
            Path to fasta file
        v_proposal : numpy.ndarray or list
            v representation of the current tree
        label_mapping : Dict[int, str]
            Current mapping of leaf labels (integer) to taxa

        Returns
        -------
        v_shuffled : numpy.ndarray
            v representation of the tree after the hill-climbing step
        current_best_loss : float
            Current best loss after the hill-climbing step
        label_mapping : Dict[int, str]
            Updated mapping of leaf labels (integer) to taxa
        """
        # Reorder v
        v_shuffled, label_mapping = reorder_v("birth_death", v, label_mapping)

        # Get current loss
        current_best_loss = raxml_loss(
            v=v_shuffled,
            label_mapping=label_mapping,
            fasta_path=fasta_path,
            model=self.model,
            tree_folder_path=self.tree_folder_path,
            outfile=f"{Path(fasta_path).stem}_best.tree",
            **self.loss_kwargs,
        )

        if self.verbose:
            print(f"Start step: {current_best_loss:.3f}")

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

                    if self.verbose:
                        grad_propose = proposal_losses[grad_choice] - current_best_loss
                        print(
                            f"Loss: {proposal_losses[grad_choice]:.3f} (diff: {grad_propose:.3f})"
                        )

                    # Update best loss
                    current_best_loss = proposal_losses[grad_choice]

        if self.verbose:
            print(f"End step: {current_best_loss:.3f}")

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
                model=self.model,
                outfile=f"{Path(fasta_path).stem}_tree{i}{j}.tree",
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


# This is a compatibility alias for the HillClimbing class.
HillClimbingOptimizer = HillClimbing
