# Change Point Detection in Hadamard Spaces by Alternating Minimization

Install with

```bash
python -m pip install -e .
```

# Usage example

```python
from hop import HOPSPD

algo = HOPSPD(penalty=0, init_cluster_centers=init_cluster_centers).fit(signal)
approx, state_sequence = algo.predict(signal, return_state_sequence=True)
```