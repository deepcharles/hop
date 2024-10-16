import numpy as np
from numba import njit


@njit
def get_full_transition_mat(n_states: int, penalty: float) -> np.ndarray:
    transition_penalty_mat = np.full((n_states, n_states), penalty, dtype=np.float64)
    for i in range(n_states):
        transition_penalty_mat[i, i] = 0.0
    return transition_penalty_mat


@njit
def get_cyclic_transition_mat(n_states: int, penalty: float) -> np.ndarray:
    transition_penalty_mat = np.full((n_states, n_states), np.inf, dtype=np.float64)
    for i in range(n_states):
        transition_penalty_mat[i, i] = 0.0
    for i in range(n_states - 1):
        transition_penalty_mat[i, i + 1] = penalty
    transition_penalty_mat[n_states - 1, 0] = penalty
    return transition_penalty_mat
