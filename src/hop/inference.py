from typing import Tuple

import numpy as np
from numba import njit


@njit
def min_plus_matvec(M: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform the min-plus matrix-vector multiplication.

    Args:
        M (np.ndarray): A 2D array of shape (n, n) representing the matrix.
        v (np.ndarray): A 1D array of shape (n,) representing the vector.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - A 1D array of shape (n,) containing the results of the min-plus multiplication.
            - A 1D array of shape (n,) containing the indices of the minimum values.
    """
    n = M.shape[0]
    out = np.empty((n,), dtype=np.float64)
    arg_out = np.empty((n,), dtype=np.int64)
    for k_row in range(n):
        arg_minimum = np.argmin(M.T[k_row] + v)
        out[k_row] = M.T[k_row, arg_minimum] + v[arg_minimum]
        arg_out[k_row] = arg_minimum
    return out, arg_out


@njit
def get_state_sequence(costs: np.ndarray, transition_mat: np.ndarray) -> np.ndarray:
    """
    Compute the optimal state sequence for the given costs and transition matrix using dynamic programming.

    Args:
        costs (np.ndarray): A 2D array of shape (n_samples, n_states) representing the cost matrix.
        transition_mat (np.ndarray): A 2D array of shape (n_states, n_states) representing the transition matrix.

    Returns:
        np.ndarray: A 1D array of shape (n_samples,) containing the optimal state sequence.
    """
    n_samples, n_states = costs.shape
    soc_array = np.empty((n_samples + 1, n_states), dtype=np.float64)
    state_array = np.empty((n_samples + 1, n_states), dtype=np.int64)
    soc_array[0] = 0
    state_array[0] = -1

    # Forward loop
    for end in range(1, n_samples + 1):
        soc_vec, state_vec = min_plus_matvec(M=transition_mat, v=soc_array[end - 1])
        soc_array[end] = soc_vec + costs[end - 1]
        state_array[end] = state_vec

    # Backtracking
    end = n_samples
    state = np.argmin(soc_array[end])
    optimal_state_sequence = np.empty(n_samples, dtype=np.int64)
    while end > 0:
        optimal_state_sequence[end - 1] = state
        state = state_array[end, state]
        end -= 1
    return optimal_state_sequence


@njit
def get_state_sequence_fixed_k(costs: np.ndarray, n_bkps_max: int = 10) -> np.ndarray:
    # costs, shape (n_samples, n_states)
    # output (optimal_state_sequence), shape (n_bkps_max + 1, n_samples)
    # the output contains the optimal state sequences for each number of changes ranging
    # from 0 to n_bkps_max
    n_samples, n_states = costs.shape
    soc_array = np.empty((n_bkps_max + 1, n_samples + 1, n_states), dtype=np.float64)
    state_array = np.empty((n_bkps_max + 1, n_samples + 1, n_states), dtype=np.int64)

    # k, t, k_state
    soc_array[:, 0, :] = 0
    state_array[:, 0, :] = -1
    for k_state in range(n_states):
        soc_array[0, 1:, k_state] = np.cumsum(costs[:, k_state])

    # Forward loop
    for end in range(1, n_samples + 1):
        for n_bkps in range(1, n_bkps_max + 1):
            for k_state in range(n_states):
                best_state = k_state
                best_soc = soc_array[n_bkps, end - 1, best_state]
                for l_state in range(n_states):
                    if (l_state != k_state) and (
                        soc_array[n_bkps - 1, end - 1, l_state] < best_soc
                    ):
                        best_state = l_state
                        best_soc = soc_array[n_bkps - 1, end - 1, l_state]
                cost_at_point = costs[end - 1, k_state]
                soc_array[n_bkps, end, k_state] = best_soc + cost_at_point
                state_array[n_bkps, end, k_state] = best_state

    # Backtracking
    optimal_state_sequence = np.empty((n_bkps_max + 1, n_samples), dtype=np.int64)
    for n_bkps in range(n_bkps_max + 1):
        end = n_samples
        state = np.argmin(soc_array[n_bkps, end])
        n_bkps_left = n_bkps
        while end > 0:
            if n_bkps_left == 0:
                optimal_state_sequence[n_bkps, :end] = state
                break
            optimal_state_sequence[n_bkps, end - 1] = state
            new_state = state_array[n_bkps_left, end, state]
            if new_state != state:
                n_bkps_left -= 1
            end -= 1
            state = new_state

    return optimal_state_sequence
