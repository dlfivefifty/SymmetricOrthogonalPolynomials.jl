using NumericalRepresentationTheory, DynamicPolynomials, Permutations

spechtpolynomial(yt::YoungMatrix, 𝐱) = prod(prod(prod(𝐱[k]-𝐱[ℓ] for ℓ=k+1:yt.columns[j]) for k = 1:yt.columns[j]-1; init=1) for j = 1:size(yt,2))
spechtpolynomial(yt::YoungTableau, 𝐱) = spechtpolynomial(YoungMatrix(yt), 𝐱)

𝐱 = [0.1,0.2,0.3,0.4,0.5]
yt = youngtableaux(Partition(3,1,1))[1]

@test spechtpolynomial(only(youngtableaux(Partition(1,1,1,1,1))), 𝐱) ≈ -spechtpolynomial(only(youngtableaux(Partition(1,1,1,1,1))), [𝐱[2]; 𝐱[1]; 𝐱[3:end]])

yms = YoungMatrix.(youngtableaux(Partition(3,1,1)))

sign(yms[1])


sign(yt)
@test spechtpolynomial(yt)

yt = YoungMatrix()
k = j =1

YoungMatrix.(youngtableaux(Partition(3,1,1)))[1].columns[3]

colsupport(YoungMatrix.(youngtableaux(Partition(3,2,1)))[1],1)

# n = 1
n = 1
@polyvar x[1:n]
λ₁ = Partition(1)
1


# n = 2
n = 2
@polyvar x[1:n]
λ₂ = Partition(2)
λ₁₊₁ = Partition(1,1)
ρ₂ = Representation(2)
ρ₁₊₁ = Representation(1,1)

τ₁ = Permutation([[1,2], fill.(3:n, n-2)...])

p₂ = 1
@test subs(p₂, x[2]=>x[1], x[1]=>x[2]) == p₂
p₁₊₁ = [x[2]-x[1]]
@test subs(p₁₊₁, x[2]=>x[1], x[1]=>x[2]) == -p₁₊₁


# n = 3
n = 3
@polyvar x[1:n]
λ₃ = Partition(3)
λ₂₊₁ = Partition(2,1)
λ₁₊₁₊₁ = Partition(1,1,1)

ρ₃ = Representation(3)
ρ₂₊₁ = Representation(2,1)
ρ₁₊₁₊₁ = Representation(1,1,1)

τ₁ = Permutation([[1,2], fill.(3:n, n-2)...])
τ₂ = Permutation([[1], [2,3], fill.(4:n, n-2)...])

p₃ = [1]
@test subs(p₃, x[2]=>x[1], x[1]=>x[2]) == ρ₃(τ₁)*p₃
@test subs(p₃, x[3]=>x[2], x[2]=>x[3]) == ρ₃(τ₂)*p₃


# classical
p₂₊₁ = [x[2]-x[1],x[3]-x[1]]
@test subs(p₂₊₁, x[2]=>x[1], x[1]=>x[2]) == [-1 0; -1 1] * p₂₊₁
@test subs(p₂₊₁, x[2]=>x[3], x[3]=>x[2]) == [0 1; 1 0] * p₂₊₁

V = [1 0;
    1/2 1] * Diagonal([2/sqrt(3),1])
@test [-1 0; -1 1]*V ≈ V*[-1 0; 0 1]
@test V\[-1 0; -1 1]*V ≈ ρ₂₊₁(τ₁)
@test V\[0 1; 1 0]*V ≈ ρ₂₊₁(τ₂)

# orthogonal

q₂₊₁ =Diagonal([sqrt(3)/2, x[3]-x[1]/2-x[2]/2]) * [p₁₊₁..., p₂]
@test all(q₂₊₁ .≈ [sqrt(3)/2 * (x[2]-x[1]);
        x[3]-x[1]/2-x[2]/2])
@test all(q₂₊₁ .≈ V\p₂₊₁)
@test all(subs(q₂₊₁, x[2]=>x[1], x[1]=>x[2]) .≈ 
            Diagonal([sqrt(3)/2, x[3]-x[1]/2-x[2]/2]) * subs([p₁₊₁..., p₂], x[2]=>x[1], x[1]=>x[2]) .≈
            Diagonal([sqrt(3)/2, x[3]-x[1]/2-x[2]/2]) * blockdiag(ρ₁₊₁(τ₁), ρ₂(τ₁)) * [p₁₊₁..., p₂] .≈
            ρ₂₊₁(τ₁)*q₂₊₁)
@test all(subs(q₂₊₁, x[2]=>x[3], x[3]=>x[2]) .≈ ρ₂₊₁(τ₂)*q₂₊₁)




p₁₊₁₊₁ = p₁₊₁ * (x[1]-x[3]) * (x[2]-x[3])
@test subs(p₁₊₁₊₁, x[2]=>x[1], x[1]=>x[2]) == ρ₁₊₁₊₁(τ₁)*p₁₊₁₊₁
@test subs(p₁₊₁₊₁, x[3]=>x[2], x[2]=>x[3]) == ρ₁₊₁₊₁(τ₂)*p₁₊₁₊₁