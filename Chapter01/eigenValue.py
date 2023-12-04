import numpy as onp
import jax.numpy as jnp

# Define matrix A
A = jnp.array([[2, -1],
              [1, 3]], dtype=jnp.float32 )
# dtype=jnp.float32： JAX use floating-point types for matrix operations

# Calculate eigenvalues
eigenvalues = jnp.linalg.eigvals(A)

print("Eigenvalues of matrix A:")
print(eigenvalues)