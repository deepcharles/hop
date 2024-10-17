from typing import Tuple

import numpy as np
from numba import njit
from sklearn.base import BaseEstimator

from hop.inference import get_state_sequence, get_state_sequence_fixed_k
from hop.transition_matrix import get_cyclic_transition_mat, get_full_transition_mat


@njit
def mean_axis_0(arr: np.ndarray) -> np.ndarray:
    """Compute the mean of each column in a 2D array.

    Args:
        arr (np.ndarray): A 2D array of shape (n_rows, n_cols) for which to
            compute the column means.

    Returns:
        np.ndarray: A 1D array of shape (n_cols,) containing the mean of each
            column.

    Raises:
        ValueError: If the input array is not 2D.

    Notes:
        This function is optimized for performance using Numba's `njit`
            decorator.
    """
    # Ensure the input is a 2D array
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Get the shape of the array
    n_rows, n_cols = arr.shape

    # Initialize the array for storing the means
    means = np.zeros(n_cols, dtype=np.float64)

    # Compute the mean for each column
    for col in range(n_cols):
        col_sum = 0.0
        for row in range(n_rows):
            col_sum += arr[row, col]
        means[col] = col_sum / n_rows

    return means


@njit
def squared_euclidean_distances(
    signal: np.ndarray, centroids: np.ndarray, out=None
) -> np.ndarray:
    """Compute the squared Euclidean distances between each sample in a signal
    array and each centroid.

    Args:
        signal (np.ndarray): A 2D array of shape (n_samples, n_dims) where each
            row represents a sample.
        centroids (np.ndarray): A 2D array of shape (n_centroids, n_dims) where
            each row represents a centroid.
        out (np.ndarray, optional): An optional 2D output array of shape
            (n_samples, n_centroids) to store the distances. If not provided, a
            new array is created.

    Returns:
        np.ndarray: A 2D array of shape (n_samples, n_centroids) where the
            element at [i, j] represents the squared Euclidean distance between
            the i-th sample and the j-th centroid.

    Notes:
        This function is optimized for performance using Numba's `njit`
            decorator.
    """
    n_samples, n_dims = np.shape(signal)
    n_centroids, _ = centroids.shape
    if out is None:
        out = np.empty((n_samples, n_centroids), dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_centroids):
            sum_sq = 0.0
            for d in range(n_dims):
                diff = signal[i, d] - centroids[j, d]
                sum_sq += diff * diff
            out[i, j] = sum_sq
    return out


@njit
def init_centroids(signal: np.ndarray, n_states: int) -> np.ndarray:
    """
    Initialize centroids for HOPL2.

    Args:
        signal (np.ndarray): A 2D array of shape (n_samples, n_dims)
            representing the input data.
        n_states (int): The number of centroids (clusters) to initialize.

    Returns:
        np.ndarray: A 2D array of shape (n_states, n_dims) representing the
            initialized centroids.
    """
    n_dims = signal.shape[1]
    centroids = np.empty(shape=(n_states, n_dims), dtype=np.float64)
    chunks = np.array_split(signal, n_states)
    for k_state in range(n_states):
        centroids[k_state] = mean_axis_0(chunks[k_state])
    return centroids


class HOPL2(BaseEstimator):
    """Change point detection with the HOP algorithm.

    Uses the Euclidean distance.
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
        """Initialize the HOPL2 algorithm.

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
        self.is_cyclic = is_cyclic
        if self.penalty is not None:
            if self.is_cyclic:
                self.transition_mat = get_cyclic_transition_mat(
                    n_states=self.n_states, penalty=self.penalty
                )
            else:
                self.transition_mat = get_full_transition_mat(
                    n_states=self.n_states, penalty=self.penalty
                )
        self.n_bkps = n_bkps
        self.max_iter = max_iter

    def fit(self, signal: np.ndarray, y=None, **kwargs):
        """Fit the HOPL2 model to the given data.

        Args:
            signal (np.ndarray): A 2D array of shape (n_samples, n_dims)
                representing the input data.
            y (np.ndarray, optional): Not used in this implementation.

        Returns:
            self: The fitted HOPL2 model.
        """
        # init
        if self.init_cluster_centers is None:
            self.cluster_centers_ = init_centroids(
                signal=signal,
                n_states=self.n_states,
            )
        else:
            self.cluster_centers_ = np.copy(self.init_cluster_centers)

        # repeat the alternate minimization
        for _ in range(self.max_iter):
            # alternate minimization

            # 1. find the best state sequence
            sq_dist_to_centers = squared_euclidean_distances(
                signal, self.cluster_centers_
            )  # shape (n_samples, n_states)
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
                self.cluster_centers_[state] = mean_axis_0(
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
            signal (np.ndarray): A 2D array of shape (n_samples, n_dims)
                representing the input data.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, list]: A tuple containing:
                - A 2D array of shape (n_samples, n_dims) representing the
                    approximated signal.
                - A 2D array of shape (n_centroids, n_dims) representing the
                    centroids.
                - A 1D array of shape (n_samples,) representing the optimal
                    state sequence.
                - A list of change points indicating the indices of the change
                    points in the signal.
        """
        sq_dist_to_centers = squared_euclidean_distances(
            signal, self.cluster_centers_
        )  # shape (n_samples, n_states)

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
