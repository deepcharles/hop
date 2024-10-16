import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp
from jax.tree_util import Partial


@jit
def fill_diagonal(a: jnp.ndarray, val: float) -> jnp.ndarray:
    """Fills the diagonal of a 2D or higher-dimensional array with a specified value.

    Args:
        a (jax.numpy.ndarray): Input array. Must have at least 2 dimensions.
        val (float): Value to fill the diagonal with.

    Returns:
        jax.numpy.ndarray: Array with diagonal filled with the specified value.

    Raises:
        AssertionError: If the input array has fewer than 2 dimensions.

    Example:
        >>> a = jnp.zeros((3, 3))
        >>> filled = fill_diagonal(a, 5)
        >>> filled
        DeviceArray([[5., 0., 0.],
                     [0., 5., 0.],
                     [0., 0., 5.]], dtype=float32)
    """
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
    return a.at[..., i, j].set(val)
