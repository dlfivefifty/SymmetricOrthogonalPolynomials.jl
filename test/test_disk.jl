using MultivariateOrthogonalPolynomials, StaticArrays, Test
using ClassicalOrthogonalPolynomials: expand

ρ = θ -> [cos(θ) -sin(θ); sin(θ) cos(θ)]

function anglevec(𝐱)
    x,y = 𝐱
    [[x,y]/sqrt(x^2+y^2) [-y,x] /sqrt(x^2+y^2)]
end

ρ̄ = (𝐱, θ) -> anglevec(ρ(θ)*𝐱) * inv(anglevec(𝐱))

Z = Zernike()

𝕍 = (n,x,y) -> hcat([[Z[SVector(x,y),Block(n+1)[k+1]], 0] for k=0:n]..., [[0,Z[SVector(x,y),Block(n+1)[k+1]]] for k=0:n]...)



θ = 0.1
x,y = 0.1,0.2




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


sum(expand(Zernike(), 𝐱 -> 1))

V = [ (x,y) -> [1,0], (x,y) -> [0,1], # m = 1

      (x,y) -> [x,y], # m = 0
      (x,y) -> [-y,x], # m = 0
      (x,y) -> [x,-y], (x,y) -> [y,x], # m = 2

      (x,y) -> legendrep(1,2*(x^2+y^2)-1)*[1,0], (x,y) -> legendrep(1,2*(x^2+y^2)-1)*[0,1], # m = 1
      (x,y) -> [(x^2-y^2)/2 x*y; x*y (y^2-x^2)/2]*[1,0], (x,y) -> [(x^2-y^2)/2 x*y; x*y (y^2-x^2)/2]*[0,1], # m=1
      ]

x,y = 0.1,0.2
@test [V[1]((ρ(θ) * [x,y])...) V[2]((ρ(θ) * [x,y])...)] ≈ ρ̄([x,y],θ) * [V[1](x,y) V[2](x,y)] * [cos(θ) sin(θ); -sin(θ) cos(θ)]

@test V[3]((ρ(θ) * [x,y])...) ≈ ρ̄([x,y],θ) * V[3](x,y)
@test V[4]((ρ(θ) * [x,y])...) ≈ ρ̄([x,y],θ) * V[4](x,y)
@test [V[5]((ρ(θ) * [x,y])...) V[6]((ρ(θ) * [x,y])...)] ≈ ρ̄([x,y],θ) * [V[5](x,y) V[6](x,y)] * [cos(2θ) sin(2θ); -sin(2θ) cos(2θ)]

@test [V[7]((ρ(θ) * [x,y])...) V[8]((ρ(θ) * [x,y])...)] ≈ ρ̄([x,y],θ) * [V[7](x,y) V[8](x,y)] * [cos(θ) sin(θ); -sin(θ) cos(θ)]
@test [V[9]((ρ(θ) * [x,y])...) V[10]((ρ(θ) * [x,y])...)] ≈ ρ̄([x,y],θ) * [V[9](x,y) V[10](x,y)] * [cos(θ) sin(θ); -sin(θ) cos(θ)]

M = (x,y) -> [chebyshevu(2,x)/4 x*y; x*y chebyshevu(2,y)/4]
@test M((ρ(θ)*[x,y])...) * ρ̄([x,y],θ) ≈ ρ̄([x,y],θ) * M(x,y)

sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[1](x,y), M(x,y)*V[1](x,y)))))

sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[1](x,y), V[2](x,y)))))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[3](x,y), V[1](x,y)))))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[4](x,y), V[1](x,y)))))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[5](x,y), V[6](x,y)))))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[7](x,y), V[6](x,y)))))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[8](x,y), V[6](x,y)))))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[9](x,y), V[8](x,y)))))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[10](x,y), V[9](x,y)))))



P = Zernike()
𝐱 = 

sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[9](x,y), V[7](x,y)))))


x,y = 0.1,0.2



r = sqrt(x^2+y^2)
θ = atan(y,x)
@test [chebyshevu(2,x)/4 x*y; x*y chebyshevu(2,y)/4] ≈ ρ̄([r,0],θ)*[chebyshevu(2,r)/4 0; 0 chebyshevu(2,0)/4]*ρ̄([x,y],-θ)

sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; [0,1]'*[chebyshevu(2,x)/4 x*y; x*y chebyshevu(2,y)/4]*[0,1])))

A = 𝐱 ->  ((x,y) = 𝐱; [chebyshevu(2,x)/4 x*y; x*y chebyshevu(2,y)/4])
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; r = norm(𝐱); θ = atan(y,x); [0,1]'*A(ρ(θ)*[r,0])*[0,1])))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; r = norm(𝐱); θ = atan(y,x); [0,1]'*ρ̄([r,0],θ)*A([r,0])*ρ̄([x,y],-θ)*[0,1])))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; r = norm(𝐱); θ = atan(y,x); [0,1]'*ρ̄([r,0],θ)*A([r,0])*ρ(-θ)*[0,1])))
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; r = norm(𝐱); θ = atan(y,x); [0,1]'*ρ(θ)*A([r,0])*ρ(-θ)*[0,1])))



sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot([y^3,0],legendrep(2,2*(x^2+y^2)-1)*[1,0]))))


sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot([chebyshevu(2,x)/4,x*y],legendrep(1,2*(x^2+y^2)-1)*[1,0]))))

sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot([chebyshevu(2,x)/4,x*y],legendrep(1,2*(x^2+y^2)-1)*[0,1]))))

@test sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(legendrep(1,2*(x^2+y^2)-1)*[1,0],legendrep(1,2*(x^2+y^2)-1)*[1,0])))) ≈ π/3
sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot([chebyshevu(2,x)/4  - 1/4*legendrep(1,2*(x^2+y^2)-1),x*y],legendrep(1,2*(x^2+y^2)-1)*[1,0]))))