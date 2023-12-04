import numpy as onp
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve
# from  jax.scipy.linalg import *
# Define matrix A and vector b
A = jnp.array([[1, 6, 0, 3],
              [0, 2, 0, 4],
              [-1, -6, 3, -3], 
              [0, -2, 0, 8]], dtype=jnp.float32)
b = jnp.array([[5],[4],[5],[4]], dtype = jnp.float32)

# Perform LU decomposition and factorization of A
# piv：permutation matrix
lu, piv = lu_factor(A)

# Solve Ax = b using LU decomposition
x = lu_solve((lu, piv), b)

print("Matrix A:")
print(A)

print("\nVector b:")
print(b)

print("\nSolution vector x:")
print(x)