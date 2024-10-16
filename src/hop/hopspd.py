from typing import Tuple

import numpy as np
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.learning.frechet_mean import FrechetMean
from sklearn.base import BaseEstimator

from hop.inference import get_state_sequence, get_state_sequence_fixed_k
from hop.transition_matrix import get_cyclic_transition_mat, get_full_transition_mat


def init_centroids(signal: np.ndarray, n_states: int) -> np.ndarray:
    """
    Initialize centroids for HOPSPD.

    Args:
        signal (np.ndarray): A 3D array of shape (n_samples, n_dims, n_dims)
            representing the input data.
        n_states (int): The number of centroids (clusters) to initialize.

    Returns:
        np.ndarray: A 3D array of shape (n_states, n_dims, n_dims) representing the
            initialized centroids.
    """
    n_dims = signal.shape[1]
    manifold = SPDMatrices(n_dims)
    mean_estimator = FrechetMean(manifold)
    centroids = np.empty(shape=(n_states, n_dims, n_dims), dtype=np.float64)
    chunks = np.array_split(signal, n_states)
    for k_state in range(n_states):
        centroids[k_state] = mean_estimator.fit(chunks[k_state]).estimate_
    return centroids


class HOPSPD(BaseEstimator):
    """Change point detection with the HOP algorithm.

    Uses the affine-invariant metric for SPD matrices.
    """

    def __init__(
        self,
        n_states: int = None,
        n_bkps: int = 1,
        penalty: float = None,
        max_iter: int = 5,
        init_cluster_centers: np.array = None,
        is_cyclic: bool = False,
    ):
        """Initialize the HOPSPD algorithm.

        Parameters:
            n_states (int, optional): The number of states (clusters) to
                initialize. Defaults to None.
            n_bkps (int, optional): The maximum number of change points to
                allow. Defaults to None.
            penalty (float, optional): The penalty value for the transition
                matrix. Defaults to None.
            max_iter (int, optional): The maximum number of iterations for the
                HOP algorithm. Defaults to 5.
            init_cluster_centers (np.array, optional): Initial cluster centers,
                shape (n_states, n_dims). Defaults to None.
            is_cyclic (bool, optional): if true, try to find a cyclic state
                sequence. Defaults to False.
        """
        err_msg = "At least 'penalty' or 'n_bkps' must be provided."
        assert (penalty is not None) or (n_bkps is not None), err_msg

        if init_cluster_centers is None:
            self.n_states = n_states
            self.init_cluster_centers = None
        else:
            self.init_cluster_centers = init_cluster_centers
            self.n_states = init_cluster_centers.shape[0]

        self.penalty = penalty
        if self.penalty is not None:
            if is_cyclic:
                self.transition_mat = get_cyclic_transition_mat(
                    n_states=self.n_states, penalty=self.penalty
                )
            else:
                self.transition_mat = get_full_transition_mat(
                    n_states=self.n_states, penalty=self.penalty
                )
        self.n_bkps = n_bkps
        self.max_iter = max_iter

    def estimate_mean(self, signal) -> np.ndarray:
        # Only call after fit
        # signal, shape (n_samples, n_dims, n_dims)
        # out,  shape (n_dims, n_dims)
        return self.mean_estimator_.fit(signal).estimate_

    def fit(self, signal: np.ndarray, y=None, **kwargs):
        # init
        n_samples, n_dims, _ = signal.shape
        self.manifold_ = SPDMatrices(n_dims)
        self.mean_estimator_ = FrechetMean(self.manifold_)
        if self.init_cluster_centers is None:
            self.cluster_centers_ = init_centroids(
                signal=signal, n_states=self.n_states
            )
        else:
            self.cluster_centers_ = self.init_cluster_centers
        sq_dist_to_centers = np.empty((n_samples, self.n_states))

        # repeat the alternate minimization
        for _ in range(self.max_iter):
            # alternate minimization

            # 1. find the best state sequence
            for k_center in range(self.n_states):
                center = self.cluster_centers_[k_center]
                sq_dist_to_centers[:, k_center] = self.manifold_.metric.squared_dist(
                    center, signal
                )
            # shape (n_samples, n_states)
            if self.penalty is not None:
                optimal_state_sequence = get_state_sequence(
                    costs=sq_dist_to_centers, transition_mat=self.transition_mat
                )  # shape (n_samples,)
            elif self.n_bkps is not None:
                all_optimal_state_sequences = get_state_sequence_fixed_k(
                    costs=sq_dist_to_centers, n_bkps_max=self.n_bkps
                )  # shape (n_bkps, n_samples)
                optimal_state_sequence = all_optimal_state_sequences[-1]

            # 2. update the centroids
            states = np.unique(optimal_state_sequence)
            for state in states:
                self.cluster_centers_[state] = self.estimate_mean(
                    signal[optimal_state_sequence == state]
                )
        self.is_fitted_ = True
        return self

    def predict(
        self,
        signal: np.ndarray,
        return_state_sequence=False,
        return_bkps=False,
        *args,
        **kwargs
    ):
        """
        Perform change point detection (CPD) on the given signal.

        Args:
            signal (np.ndarray): A 2D array of shape (n_samples, n_dims) representing the input data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, list]: A tuple containing:
                - A 2D array of shape (n_samples, n_dims) representing the approximated signal.
                - A 2D array of shape (n_centroids, n_dims) representing the centroids.
                - A 1D array of shape (n_samples,) representing the optimal state sequence.
                - A list of change points indicating the indices of the change points in the signal.
        """
        n_samples = signal.shape[0]
        sq_dist_to_centers = np.empty((n_samples, self.n_states))
        for k_center in range(self.n_states):
            center = self.cluster_centers_[k_center]
            sq_dist_to_centers[:, k_center] = self.manifold_.metric.squared_dist(
                center, signal
            )
            # shape (n_samples, n_states)

        if self.penalty is not None:
            optimal_state_sequence = get_state_sequence(
                costs=sq_dist_to_centers, transition_mat=self.transition_mat
            )  # shape (n_samples,)
        elif self.n_bkps is not None:
            all_optimal_state_sequences = get_state_sequence_fixed_k(
                costs=sq_dist_to_centers, n_bkps_max=self.n_bkps
            )  # shape (n_bkps, n_samples)
            optimal_state_sequence = all_optimal_state_sequences[-1]

        approx = self.cluster_centers_[optimal_state_sequence]
        (bkps,) = np.nonzero(np.diff(optimal_state_sequence))
        bkps = bkps.tolist() + [signal.shape[0]]

        if return_state_sequence and return_bkps:
            return approx, optimal_state_sequence, bkps
        if return_state_sequence:
            return approx, optimal_state_sequence
        if return_bkps:
            return approx, bkps
        return approx
