using MultivariateOrthogonalPolynomials, StaticArrays, Test

Z = Zernike()

𝕍 = (n,x,y) -> hcat([[Z[SVector(x,y),Block(n+1)[k+1]], 0] for k=0:n]..., [[0,Z[SVector(x,y),Block(n+1)[k+1]]] for k=0:n]...)


ρ = θ -> [cos(θ) -sin(θ); sin(θ) cos(θ)]
θ = 0.1
x,y = 0.1,0.2


function anglevec(xy)
    x,y = xy
    [[x,y]/sqrt(x^2+y^2) [-y,x] /sqrt(x^2+y^2)]
end

ρ̄ = (xy, θ) -> anglevec(ρ(θ)*xy) * inv(anglevec(xy))

@test  𝕍(0,ρ(θ)*[x,y]...) ≈ ρ̄([x,y], θ) * 𝕍(0,x,y) * ρ(-θ)


ρ̄([1,0],π/4) * [1,0]

n = 1; x1,y1 = 0.2,0.3; x2,y2 = -0.123,0.245;
C = [ρ̄([x,y], θ)*𝕍(n,x,y);
    ρ̄([x1,y1], θ)*𝕍(n,x1,y1);
    ρ̄([x2,y2], θ)*𝕍(n,x2,y2)] \ [𝕍(n,ρ(θ)*[x,y]...); 𝕍(n,ρ(θ)*[x1,y1]...); 𝕍(n,ρ(θ)*[x2,y2]...)]

eigvals(C) # m = 0 (2x), m = ±2

n = 2; x1,y1 = 0.2,0.3; x2,y2 = -0.123,0.245;
C = [ρ̄([x,y], θ)*𝕍(n,x,y);
    ρ̄([x1,y1], θ)*𝕍(n,x1,y1);
    ρ̄([x2,y2], θ)*𝕍(n,x2,y2)] \ [𝕍(n,ρ(θ)*[x,y]...); 𝕍(n,ρ(θ)*[x1,y1]...); 𝕍(n,ρ(θ)*[x2,y2]...)]

eigvals(C) # m = ±1 (2x), m = ±3



n = 3; x1,y1 = 0.2,0.3; x2,y2 = -0.123,0.245; x3,y3 = 0.421, -0.9
C = [ρ̄([x,y], θ)*𝕍(n,x,y);
    ρ̄([x1,y1], θ)*𝕍(n,x1,y1);
    ρ̄([x2,y2], θ)*𝕍(n,x2,y2);
    ρ̄([x3,y3], θ)*𝕍(n,x3,y3)] \ [𝕍(n,ρ(θ)*[x,y]...); 𝕍(n,ρ(θ)*[x1,y1]...); 𝕍(n,ρ(θ)*[x2,y2]...); 𝕍(n,ρ(θ)*[x3,y3]...)]

eigvals(C) # m = 0 (2x), m = ±2 (2x), m = ±4

# rotationally invariant matrices
# (A * v)(ρ*x) == ρ̄ * A*v
# (A(ρ*x) * ρ̄ v) == ρ̄ * A*v
# A(ρ*x) * ρ̄ == ρ̄ * A

Q = eigen(ρ̄([x,y], θ)).vectors

Q'*ρ̄([x,y], θ)*Q

A = Q*Diagonal(randn(2))*Q'

θ=0.3; @test A*ρ̄([x,y], θ) ≈ ρ̄([x,y], θ)*A

# r^m * exp(im*m*θ) * f(r^2)
# ∇  =  r^(m-1)exp(im*m*θ) * ((m*f(r^2) + 2r^2*f'(r^2))e_r + im * m * f(r^2) *e_θ)
# ∇⟂ =  r^(m-1)exp(im*m*θ) * ((m*f(r^2) + 2r^2*f'(r^2))e_r + im * m * f(r^2) *e_θ)

# m= 0: r*e_r = [x,y]
# m= 1: x * [x,y], y * [-y,x]

# r^(m+1) * exp(im*m*θ) * f(r^2) * e_r is degree (f + m+1)

# [x^2 x*y; y*x y^2] 
# [1 0; 0 1]

A = (x,y) -> [x^2 x*y; y*x y^2] 

@test A(ρ(θ)*[x,y]...)*ρ̄([x,y], θ) ≈ ρ̄([x,y], θ)*A(x,y)
