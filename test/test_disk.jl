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

n = 1
C = 𝕍(n,x,y) \  ρ̄(ρ(θ)*[x,y], -θ) * 𝕍(n,ρ(θ)*[x,y]...)

θ = 0.1; A = (ρ̄([x,y], θ) * 𝕍(n,x,y)) \ 𝕍(n,ρ(θ)*[x,y]...)
θ = 0.2; B = (ρ̄([x,y], θ) * 𝕍(n,x,y)) \ 𝕍(n,ρ(θ)*[x,y]...)
θ = 0.3; C = (ρ̄([x,y], θ) * 𝕍(n,x,y)) \ 𝕍(n,ρ(θ)*[x,y]...)

A*B
C
eigvals(C) # m = 0 (2x), m = ±1
eigen(C)
exp(im*θ)

ρ̄(ρ(θ)*[x,y], -θ) * 𝕍(n,ρ(θ)*[x,y]...) * Q == 𝕍(n,x,y) * C

n = 2
C = 𝕍(n,x,y) \  ρ̄(ρ(θ)*[x,y], -θ) * 𝕍(n,ρ(θ)*[x,y]...)
eigvals(C) # m = 0 (4x), m = ±1

C'C
𝕍(n,x,y) * C - ρ̄(ρ(θ)*[x,y], -θ) * 𝕍(n,ρ(θ)*[x,y]...)
exp(im*θ)