using NumericalRepresentationTheory, DynamicPolynomials, Permutations


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

p₂₊₁ = [p₁₊₁...,p₂]
@test subs(p₂₊₁, x[2]=>x[1], x[1]=>x[2]) == ρ₂₊₁(τ₁)*p₂₊₁
subs(p₂₊₁, x[2]=>x[3], x[3]=>x[2])

ρ₂₊₁(τ₂)


p₁₊₁₊₁ = p₁₊₁ * (x[1]-x[3]) * (x[2]-x[3])
@test subs(p₁₊₁₊₁, x[2]=>x[1], x[1]=>x[2]) == ρ₁₊₁₊₁(τ₁)*p₁₊₁₊₁
@test subs(p₁₊₁₊₁, x[3]=>x[2], x[2]=>x[3]) == ρ₁₊₁₊₁(τ₂)*p₁₊₁₊₁