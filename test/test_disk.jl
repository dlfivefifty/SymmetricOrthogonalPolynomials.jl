using MultivariateOrthogonalPolynomials, StaticArrays, Test, ClassicalOrthogonalPolynomials
using ClassicalOrthogonalPolynomials: expand

ρ = θ -> @SMatrix([cos(θ) -sin(θ); sin(θ) cos(θ)])

function anglevec(𝐱)
    x,y = 𝐱
    @SMatrix([[x,y]/sqrt(x^2+y^2) [-y,x] /sqrt(x^2+y^2)])
end

# ρ̄ = (𝐱, θ) -> anglevec(ρ(θ)*𝐱) * inv(anglevec(𝐱))

ρ̄ = (𝐱, θ) -> ρ(θ)

Z = Zernike()

𝕍 = (n,x,y) -> hcat([[Z[SVector(x,y),Block(n+1)[k+1]], 0] for k=0:n]..., [[0,Z[SVector(x,y),Block(n+1)[k+1]]] for k=0:n]...)



θ = 0.1
𝐱 = SVector(0.1,0.2); x,y = 𝐱




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

    # d = 1, n = 3
      (x,y) -> [x,y], # m = 0
      (x,y) -> [-y,x], # m = 0
      (x,y) -> [x,-y], (x,y) -> [y,x], # m = 2
    # d = 2, n = 7

      (x,y) -> legendrep(1,2*(x^2+y^2)-1)*V[1](x,y), (x,y) -> legendrep(1,2*(x^2+y^2)-1)*V[2](x,y), # m = 1
      (x,y) -> [(x^2-y^2) 2x*y; 2x*y (y^2-x^2)]*V[1](x,y), (x,y) -> [(x^2-y^2) 2x*y; 2x*y (y^2-x^2)]*V[2](x,y), # m=1
      (x,y) -> (r = sqrt(x^2+y^2); θ = atan(y,x); m=3; m*r^(m-2)*(cos(m*θ)*[x,y] - sin(m*θ)*[-y,x])),
      (x,y) -> (r = sqrt(x^2+y^2); θ = atan(y,x); m=3; m*r^(m-2)*(sin(m*θ)*[x,y] + cos(m*θ)*[-y,x])), # m = 3

    # d = 3, n = 13
      (x,y) -> jacobip(1,0,1,2*(x^2+y^2)-1)*V[3](x,y), # m = 0
      (x,y) -> jacobip(1,0,1,2*(x^2+y^2)-1)*V[4](x,y), # m = 0
      (x,y) -> jacobip(1,0,1,2*(x^2+y^2)-1)*V[5](x,y), (x,y) -> jacobip(1,0,1,2*(x^2+y^2)-1)*V[6](x,y), # m = 2
      (x,y) -> [(x^2-y^2) 2x*y; 2x*y (y^2-x^2)]*V[5](x,y), (x,y) -> [(x^2-y^2) 2x*y; 2x*y (y^2-x^2)]*V[6](x,y), # m = 2
      (x,y) -> (r = sqrt(x^2+y^2); θ = atan(y,x); m=4; m*r^(m-2)*(cos(m*θ)*[x,y] - sin(m*θ)*[-y,x])),
      (x,y) -> (r = sqrt(x^2+y^2); θ = atan(y,x); m=4; m*r^(m-2)*(sin(m*θ)*[x,y] + cos(m*θ)*[-y,x])), # m = 4  

    # d = 4, n = 21      
        (x,y) -> legendrep(2,2*(x^2+y^2)-1)*V[1](x,y), (x,y) -> legendrep(2,2*(x^2+y^2)-1)*V[2](x,y), # m = 1
        (x,y) -> jacobip(1, 0, 2, 2*(x^2+y^2)-1)*[(x^2-y^2) 2x*y; 2x*y (y^2-x^2)]*V[1](x,y), (x,y) -> jacobip(1, 0, 2, 2*(x^2+y^2)-1)*[(x^2-y^2) 2x*y; 2x*y (y^2-x^2)]*V[2](x,y), # m=1
        (x,y) -> jacobip(1, 0, 2, 2*(x^2+y^2)-1)*V[11](x,y), (x,y) -> jacobip(1, 0, 2, 2*(x^2+y^2)-1)*V[12](x,y), # m = 3
        (x,y) -> [(x^2-y^2) 2x*y; 2x*y (y^2-x^2)]*V[11](x,y), (x,y) -> [(x^2-y^2) 2x*y; 2x*y (y^2-x^2)]*V[12](x,y), # m = 3
        (x,y) -> (r = sqrt(x^2+y^2); θ = atan(y,x); m=5; m*r^(m-2)*(cos(m*θ)*[x,y] - sin(m*θ)*[-y,x])),
      (x,y) -> (r = sqrt(x^2+y^2); θ = atan(y,x); m=5; m*r^(m-2)*(sin(m*θ)*[x,y] + cos(m*θ)*[-y,x])), # m = 5
      ]


𝐱 = SVector(0.1,0.2); x,y = 𝐱
@test [V[1]((ρ(θ) * 𝐱)...) V[2]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[1](x,y) V[2](x,y)] * ρ(-θ)

@test V[3]((ρ(θ) * 𝐱)...) ≈ ρ̄(𝐱,θ) * V[3](x,y)
@test V[4]((ρ(θ) * 𝐱)...) ≈ ρ̄(𝐱,θ) * V[4](x,y)
@test [V[5]((ρ(θ) * 𝐱)...) V[6]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[5](x,y) V[6](x,y)] *  ρ(-θ)^2

@test [V[7]((ρ(θ) * 𝐱)...) V[8]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[7](x,y) V[8](x,y)] *  ρ(-θ)
@test [V[9]((ρ(θ) * 𝐱)...) V[10]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[9](x,y) V[10](x,y)] *  ρ(-θ)
@test [V[11]((ρ(θ) * 𝐱)...) V[12]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[11](x,y) V[12](x,y)] *  ρ(-θ)^3

@test V[13]((ρ(θ) * 𝐱)...) ≈ ρ̄(𝐱,θ) * V[13](x,y)
@test V[14]((ρ(θ) * 𝐱)...) ≈ ρ̄(𝐱,θ) * V[14](x,y)
@test [V[15]((ρ(θ) * 𝐱)...) V[16]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[15](x,y) V[16](x,y)] *  ρ(-θ)^2 
@test [V[17]((ρ(θ) * 𝐱)...) V[18]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[17](x,y) V[18](x,y)] *  ρ(-θ)^2 
@test [V[19]((ρ(θ) * 𝐱)...) V[20]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[19](x,y) V[20](x,y)] *  ρ(-θ)^4

# d = 3
@test [V[21]((ρ(θ) * 𝐱)...) V[22]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[21](x,y) V[22](x,y)] *  ρ(-θ)
@test [V[23]((ρ(θ) * 𝐱)...) V[24]((ρ(θ) * 𝐱)...)] ≈ ρ̄(𝐱,θ) * [V[23](x,y) V[24](x,y)] *  ρ(-θ)


# d = 2
n = 10; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 11; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 12; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))

# d = 3
n = 13; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 14; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 15; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 16; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 17; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 18; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 19; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 20; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))

# d = 4
n = 21; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 22; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 23; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 24; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 25; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 26; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 27; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 28; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 29; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 30; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))




w = (x,y) -> jacobip(1, 0, 2, 2*(x^2+y^2)-1)*[(x^2-y^2) 2x*y; 2x*y (y^2-x^2)]*[1,0]

sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(w(x,y), [1+x+y+x^2+x*y+y^2+x^3+x^2*y+x*y^2+y^3,1+x+y+x^2+x*y+y^2+x^3+x^2*y+x*y^2+y^3]))))

###
# OLD
###


M = (x,y) -> [chebyshevu(2,x)/4 x*y; x*y chebyshevu(2,y)/4]
@test M((ρ(θ)*[x,y])...) * ρ̄([x,y],θ) ≈ ρ̄([x,y],θ) * M(x,y)

sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[1](x,y), M(x,y)*V[1](x,y)))))


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

A = (x,y) ->  [-2x*y   x^2-y^2;
               x^2-y^2 2x*y]

𝐱 = SVector(0.1,0.2)
@test A((ρ(θ)𝐱)...)ρ̄(𝐱,θ) ≈ ρ̄(𝐱,θ)A(𝐱...)



###
# matrix basis
###


A = (x,y) -> I(2)
@test A((ρ(θ)𝐱)...)ρ̄(𝐱,θ) ≈ ρ̄(𝐱,θ)A(𝐱...)
A = (x,y) -> [0 1; -1 0]
@test A((ρ(θ)𝐱)...)ρ̄(𝐱,θ) ≈ ρ̄(𝐱,θ)A(𝐱...)
A = (x,y) -> [1 0; 0 -1]
B = (x,y) -> [0 1; 1 0]
@test [ρ̄(𝐱,-θ)A((ρ(θ)𝐱)...)ρ̄(𝐱,θ),ρ̄(𝐱,-θ)B((ρ(θ)𝐱)...)ρ̄(𝐱,θ)] ≈ ρ(θ)^2 * [A(𝐱...),B(𝐱...)]



using StaticArrays
B = (x,y) -> [@SMatrix([1 0; 0 0]), @SMatrix([0 1; 0 0]), @SMatrix([0 0; 1 0]), @SMatrix([0 0 ; 0 1])]


B(𝐱...)

𝐱

using BlockArrays

θ
@test [(𝐱 = SVector(0.1,0.2); (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ))))  (𝐱 = SVector(0.1,-0.3); (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ))))] ≈
    [0.2 -0.4 -0.4 0.8;
 0.4 0.2 -0.8 -0.4;
 0.4 -0.8 0.2 -0.4;
 0.8 0.4 0.4 0.2] * [(𝐱 = SVector(0.1,0.2); B(𝐱...)) (𝐱 = SVector(0.1,-0.3); B(𝐱...))]

 R_θ = [0.2 -0.4 -0.4 0.8;
        0.4 0.2 -0.8 -0.4;
        0.4 -0.8 0.2 -0.4;
        0.8 0.4 0.4 0.2]

R_θ'R_θ

eigen(Q)




B = (x,y) -> [[(A = zero(SMatrix{2,2,Float64}); setindex(A, x, k)) for k=1:4]; [(A = zero(SMatrix{2,2,Float64}); setindex(A, y, k)) for k=1:4]]

xs = [SVector(0.1,0.2), SVector(-0.3,0.1), SVector(0.3,-0.7), SVector(-0.1,-0.1)]
M = [getindex.((𝐱 = xs[1]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),1) getindex.((𝐱 = xs[2]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),1) getindex.((𝐱 = xs[3]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),1) getindex.((𝐱 = xs[4]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),1)]'
X11 = [first.(xs) last.(xs)] \ M
M = [getindex.((𝐱 = xs[1]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),2) getindex.((𝐱 = xs[2]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),2) getindex.((𝐱 = xs[3]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),2) getindex.((𝐱 = xs[4]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),2)]'
X21 = [first.(xs) last.(xs)] \ M
M = [getindex.((𝐱 = xs[1]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),3) getindex.((𝐱 = xs[2]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),3) getindex.((𝐱 = xs[3]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),3) getindex.((𝐱 = xs[4]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),3)]'
X12 = [first.(xs) last.(xs)] \ M
M = [getindex.((𝐱 = xs[1]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),4) getindex.((𝐱 = xs[2]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),4) getindex.((𝐱 = xs[3]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),4) getindex.((𝐱 = xs[4]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),4)]'
X22 = [first.(xs) last.(xs)] \ M

Q = [X11[1,:]'; X21[1,:]'; X12[1,:]'; X22[1,:]';
        X11[2,:]'; X21[2,:]'; X12[2,:]'; X22[2,:]';]


@test (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ))) ≈ Q'*B(𝐱...)



getindex.((𝐱 = xs[1]; (Ref(ρ̄(𝐱, -θ)) .* B((ρ(θ)*𝐱)...) .* Ref(ρ̄(𝐱, θ)))),1)



V = [# d = 0
    (x,y) -> [1 0; 0 1], # m = 0
     (x,y) -> [0 -1; 1 0], # m = 0
     (x,y) -> [1 0; 0 -1], (x,y) -> [0 1; 1 0], # m = 2

     # d = 1, n = 5
     (x,y) -> [x y; y -x], (x,y) -> [-y x; x y], # m = 1
     (x,y) -> [x -y; y x], (x,y) -> [y x; -x y], # m = 1
     (x,y) -> [x y; -y x], (x,y) -> [y -x; x y], # m = 1
     (x,y) -> [x -y; -y -x], (x,y) -> [y x; x -y], # m = 3

    # d = 2, n = 13
    (x,y) -> legendrep(1,2*(x^2+y^2)-1)*V[1](x,y), # m = 0
    (x,y) -> [x^2-y^2 2x*y; 2x*y y^2-x^2], # m = 0    A(x,y)*V[1](x,y)
    (x,y) -> legendrep(1,2*(x^2+y^2)-1)*V[2](x,y), # m = 0
    (x,y) -> [2x*y y^2-x^2; y^2-x^2 -2x*y], # m = 0 .  A(x,y)*V[2](x,y)
    (x,y) -> legendrep(1,2*(x^2+y^2)-1)*V[3](x,y), (x,y) -> legendrep(1,2*(x^2+y^2)-1)*V[4](x,y), # m = 2
    (x,y) -> [x^2-y^2 -2x*y; 2x*y x^2-y^2], (x,y) -> [2x*y x^2-y^2; y^2-x^2 2x*y], # m = 2
     (x,y) ->[2x*y x^2-y^2; x^2-y^2 -2x*y],
    (x,y) ->[2x*y y^2-x^2; x^2-y^2 2x*y],
    (x,y) -> [x^2-y^2 -2x*y; -2x*y x^2-y^2],
    # (x,y) -> V[16](x,y)*V[3](x,y), (x,y) -> V[16](x,y)*V[4](x,y), # m = 2
 ]

# d = 0
@test V[1]((ρ(θ)𝐱)...)ρ̄(𝐱,θ) ≈ ρ̄(𝐱,θ)V[1](𝐱...)
@test V[2]((ρ(θ)𝐱)...)ρ̄(𝐱,θ) ≈ ρ̄(𝐱,θ)V[2](𝐱...)
@test [V[3]((ρ(θ)𝐱)...)ρ̄(𝐱,θ),V[4]((ρ(θ)𝐱)...)ρ̄(𝐱,θ)] ≈ ρ(θ)^2*[ρ̄(𝐱,θ)V[3](𝐱...),ρ̄(𝐱,θ)V[4](𝐱...)]

#d = 1
n = 5; @test [V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ),V[n+1]((ρ(θ)𝐱)...)ρ̄(𝐱,θ)] ≈ ρ(θ)*[ρ̄(𝐱,θ)V[n](𝐱...),ρ̄(𝐱,θ)V[n+1](𝐱...)]
n = 7; @test [V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ),V[n+1]((ρ(θ)𝐱)...)ρ̄(𝐱,θ)] ≈ ρ(θ)*[ρ̄(𝐱,θ)V[n](𝐱...),ρ̄(𝐱,θ)V[n+1](𝐱...)]
n = 9; @test [V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ),V[n+1]((ρ(θ)𝐱)...)ρ̄(𝐱,θ)] ≈ ρ(θ)*[ρ̄(𝐱,θ)V[n](𝐱...),ρ̄(𝐱,θ)V[n+1](𝐱...)]
n = 11; @test [V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ),V[n+1]((ρ(θ)𝐱)...)ρ̄(𝐱,θ)] ≈ ρ(θ)^3*[ρ̄(𝐱,θ)V[n](𝐱...),ρ̄(𝐱,θ)V[n+1](𝐱...)]

# d = 2
n = 13; @test V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ) ≈ ρ̄(𝐱,θ)V[n](𝐱...)
n = 14; @test V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ) ≈ ρ̄(𝐱,θ)V[n](𝐱...)
n = 15; @test V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ) ≈ ρ̄(𝐱,θ)V[n](𝐱...)
n = 16; @test V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ) ≈ ρ̄(𝐱,θ)V[n](𝐱...)
n = 17; @test [V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ),V[n+1]((ρ(θ)𝐱)...)ρ̄(𝐱,θ)] ≈ ρ(θ)^2*[ρ̄(𝐱,θ)V[n](𝐱...),ρ̄(𝐱,θ)V[n+1](𝐱...)]
n = 19; @test [V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ),V[n+1]((ρ(θ)𝐱)...)ρ̄(𝐱,θ)] ≈ ρ(θ)^2*[ρ̄(𝐱,θ)V[n](𝐱...),ρ̄(𝐱,θ)V[n+1](𝐱...)]
n = 21; @test_broken [V[n]((ρ(θ)𝐱)...)ρ̄(𝐱,θ),V[n+1]((ρ(θ)𝐱)...)ρ̄(𝐱,θ)] ≈ ρ(θ)^4*[ρ̄(𝐱,θ)V[n](𝐱...),ρ̄(𝐱,θ)V[n+1](𝐱...)]

# d = 0
n = 2; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 3; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 4; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))

# d = 1
n = 5; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 6; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 7; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 8; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 9; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 10; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 11; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 12; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))

# d= 2
n = 13; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 14; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 15; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 16; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 17; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 18; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 19; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 20; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 21; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 22; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))
n = 23; sum(expand(Zernike(), 𝐱 -> ((x,y) = 𝐱; dot(V[n](x,y), sum(V[k](x,y) for k=1:n-1)))))