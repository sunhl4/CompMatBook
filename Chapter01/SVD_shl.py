import numpy as onp
import jax.numpy as jnp
import jax
# Define matrix A
A = jnp.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=jnp.float32)

# Perform SVD
#use jax.numpy.linalg(or jax.scipy.linalg)
U, S, Vt = jnp.linalg.svd(A)

# Create the Sigma matrix with the same shape as A
Sigma = onp.zeros(A.shape)
for i in range(jnp.min(jnp.array(A.shape))):
    Sigma[i, i] = S[i]

print("Matrix A:")
print(A)

print("\nMatrix U:")
print(U)

print("\nMatrix Sigma:")
print(Sigma)

print("\nMatrix Vt:")
print(Vt)