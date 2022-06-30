using LinearAlgebra, ClassicalOrthogonalPolynomials, Test

W = Weighted(Jacobi(1,1))
P = Legendre()

𝕍 = (n,x,y) -> hcat([[P[x,n-k+1]*W[y,k+1], 0] for k=0:n]..., [[0,W[x,n-k+1]*P[y,k+1]] for k=0:n]...)

x,y=0.1,0.2;
n = 0
@test [0 1; 1 0] * 𝕍(0,y,x) ≈ 𝕍(0,x,y) * [0 1; 1 0]
@test [1 0; 0 -1] * 𝕍(0,x,-y) ≈ 𝕍(0,x,y) * [1 0; 0 -1]

n = 1
@test [0 1; 1 0] * 𝕍(n,y,x) ≈ 𝕍(n,x,y) * I(2(n+1))[end:-1:1,:]
@test [1 0; 0 -1] * 𝕍(n,x,-y) ≈ 𝕍(n,x,y) * Diagonal([(-1).^(0:n); -(-1).^(0:n)])



σ = I(2(n+1))[end:-1:1,:]
μ = Diagonal([(-1).^(0:n); -(-1).^(0:n)])

# σ * Q 

nullspace(Matrix([σ - I;  μ - I]))
nullspace(Matrix([σ - I;  μ + I]))
nullspace(Matrix([σ + I;  μ - I]))
nullspace(Matrix([σ + I;  μ + I]))

ℚ = (n,x,y) -> 𝕍(n,x,y) * [1 0 -1 0; 0 1 0 1; 0 1 0 -1; 1 0 1 0]/sqrt(2)

@test [0 1; 1 0] * ℚ(n,y,x) ≈ ℚ(n,x,y) * Diagonal([1,1,-1,-1])
@test [1 0; 0 -1] * ℚ(n,x,-y) ≈ ℚ(n,x,y) * Diagonal([1,-1,1,-1])


n = 2
@test [0 1; 1 0] * 𝕍(n,y,x) ≈ 𝕍(n,x,y) * I(2(n+1))[end:-1:1,:]
@test  [1 0; 0 -1] * 𝕍(n,x,-y) ≈ 𝕍(n,x,y) * Diagonal([(-1).^(0:n); -(-1).^(0:n)])

