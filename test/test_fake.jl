using DynamicPolynomials, NumericalRepresentationTheory, Permutations

function polypermgen(n, p)
    @polyvar x[1:n] monomial_order = Graded{DynamicPolynomials.Reverse{LexOrder}}

    μ = monomials(x, p)
    d = length(μ)

    τ = typeof(spzeros(Int,d,d))[]
    for k = 1:n-1
        push!(τ, spzeros(Int, d, d))
        σ = sortperm(subs.(μ, x[k] => x[k+1], x[k+1] => x[k]))
        setindex!.(Ref(τ[k]), 1, σ, 1:d)
    end
    Representation(τ)
end


#####
# n = 3
#####

n = 3

@polyvar x[1:n]
@polyvar y[1:n]
@polyvar z[1:n]

# 1D Fake polys
f₁ = 1
f₂ = [(x[1]-x[2])/sqrt(2), (x[1]+x[2])/sqrt(6) - sqrt(2/3) * x[3]]
f₃ = [(x[1]^2-x[2]^2)/sqrt(2), (x[1]^2+x[2]^2)/sqrt(6) - sqrt(2/3) * x[3]^2]
f₄ = (x[1]-x[2])*(x[1]-x[3])*(x[2]-x[3])


ρₜ = Representation(3)
ρ₂₊₁ = Representation(2,1)
ρₛ = Representation(1,1,1)
τ₁ = Permutation([[1,2], [3]])
c = Permutation([[1,2,3]])


@test all(subs(f₁, x[1] => x[2], x[2] => x[1]) .≈ ρₜ(τ₁) * f₁)
@test all(subs(f₁,  ([x[2:n]; x[1]] .=> x)...) .≈ ρₜ(c) * f₁)

@test all(subs(f₂, x[1] => x[2], x[2] => x[1]) .≈ ρ₂₊₁(τ₁) * f₂)
@test all(subs(f₂,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₂₊₁(c) * f₂)

@test all(subs(f₃, x[1] => x[2], x[2] => x[1]) .≈ ρ₂₊₁(τ₁) * f₃)
@test all(subs(f₃,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₂₊₁(c) * f₃)

@test subs(f₄, x[1] => x[2], x[2] => x[1]) ≈ only(ρₛ(τ₁)) * f₄
@test subs(f₄,  ([x[2:n]; x[1]] .=> x)...) .≈ only(ρₛ(c)) * f₄



# p = 0
f₁

# p = 1
@test length([(x[1]+x[2]+x[3])*f₁; f₂]) == 3

# p = 2
@test length([(x[1]^2+x[2]^2+x[3]^2)*f₁;
    (x[1]x[2]+x[1]x[3]+x[2]x[3])*f₁;
    (x[1]+x[2]+x[3])*f₂;
    f₃;
    ]) == 6

# p = 3
@test length([(x[1]^3+x[2]^3+x[3]^3)*f₁;
    (x[1]^2*x[2]+x[1]^2*x[3]+x[2]^2*x[1]+x[2]^2*x[3]+x[3]^2*x[1]+x[3]^2*x[2])*f₁;
    (x[1]x[2]x[3])*f₁;
    (x[1]^2+x[2]^2+x[3]^2)*f₂;
    (x[1]x[2]+x[1]x[3]+x[2]x[3])*f₂;
    (x[1]+x[2]+x[3])*f₃;
    f₄;
    ]) == 10



####
# 2D
####

λ,Q = blockdiagonalize(ρ₂₊₁ ⊗ ρ₂₊₁)

@test Q[:,1] ≈ [0,1/sqrt(2),-1/sqrt(2),0]

# p = 2
F₃ = f₂[2]subs(f₂[1],(x .=> y)...) - f₂[1]subs(f₂[2],(x .=> y)...)

# p = 3
F₁ = f₄ # ρₛ ⊗ ρₜ
F₂ = subs(f₄,(x .=> y)...) # ρₜ ⊗ ρₛ
F₄ = f₂[2]subs(f₃[1],(x .=> y)...) - f₂[1]subs(f₃[2],(x .=> y)...)
F₅ = f₃[2]subs(f₂[1],(x .=> y)...) - f₃[1]subs(f₂[2],(x .=> y)...)

# p = 4
F₆ = f₃[2]subs(f₃[1],(x .=> y)...) - f₃[1]subs(f₃[2],(x .=> y)...)


for F in (F₁,F₂,F₃,F₄,F₅,F₆)
    @test subs(F, x[1] => x[2], x[2] => x[1], y[1] => y[2], y[2] => y[1]) ≈ -F
    @test subs(F,  ([x[2:n]; x[1]] .=> x)..., ([y[2:n]; y[1]] .=> y)...) ≈ F
end

####
# 3D
####

@test !haskey(multiplicities(ρₜ ⊗ ρₜ ⊗ ρₜ), Partition(1,1,1))
@test multiplicities(ρₜ ⊗ ρₜ ⊗ ρₛ)[Partition(1,1,1)] == 1
@test !haskey(multiplicities(ρₜ ⊗ ρₜ ⊗ ρ₂₊₁), Partition(1,1,1))
@test !haskey(multiplicities(ρₜ ⊗ ρₛ ⊗ ρₛ), Partition(1,1,1))
@test !haskey(multiplicities(ρₜ ⊗ ρₛ ⊗ ρ₂₊₁), Partition(1,1,1))
@test multiplicities(ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1
@test multiplicities(ρₛ ⊗ ρₛ ⊗ ρₛ)[Partition(1,1,1)] == 1
@test !haskey(multiplicities(ρₛ ⊗ ρₛ ⊗ ρ₂₊₁), Partition(1,1,1))
@test multiplicities(ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1
@test multiplicities(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1

@test blockdiagonalize(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[2][:,1] == [-1,0,0,1,0,1,1,0]/2





Fs = [
# p = 2
F₃ # ρ₂₊₁ ⊗ ρₜ ⊗ ρ₂₊₁
subs(F₃,(y .=> z)...) # ρ₂₊₁ ⊗ ρₜ ⊗ ρ₂₊₁
subs(F₃,(x .=> y)..., (y .=> z)...) # ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 3
f₄ # ρₛ ⊗ ρₜ ⊗ ρₜ
subs(f₄,(x .=> y)...) # ρₜ ⊗ ρₛ ⊗ ρₜ
subs(f₄,(x .=> z)...) # ρₜ ⊗ ρₜ ⊗ ρₛ
F₄ # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₜ
subs(F₄,(y .=> z)...) # ρ₂₊₁ ⊗ ρₜ ⊗ ρ₂₊₁
subs(F₄,(x .=> y)..., (y .=> z)...) # ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
F₅ # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₜ
subs(F₅,(y .=> z)...) # ρ₂₊₁ ⊗ ρₜ ⊗ ρ₂₊₁
subs(F₅,(x .=> y)..., (y .=> z)...) # ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂[1]subs(f₂[1],(x .=> y)...)subs(f₂[1],(x .=> z)...) + f₂[1]subs(f₂[2],(x .=> y)...)subs(f₂[2],(x .=> z)...) + f₂[2]subs(f₂[1],(x .=> y)...)subs(f₂[2],(x .=> z)...) + f₂[2]subs(f₂[2],(x .=> y)...)subs(f₂[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁


# p = 4
F₆ # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₜ
subs(F₆,(y .=> z)...) # ρ₂₊₁ ⊗ ρₜ ⊗ ρ₂₊₁
subs(F₆,(x .=> y)..., (y .=> z)...) # ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₃[1]subs(f₂[1],(x .=> y)...)subs(f₂[1],(x .=> z)...) + f₃[1]subs(f₂[2],(x .=> y)...)subs(f₂[2],(x .=> z)...) + f₃[2]subs(f₂[1],(x .=> y)...)subs(f₂[2],(x .=> z)...) + f₃[2]subs(f₂[2],(x .=> y)...)subs(f₂[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂[1]subs(f₃[1],(x .=> y)...)subs(f₂[1],(x .=> z)...) + f₂[1]subs(f₃[2],(x .=> y)...)subs(f₂[2],(x .=> z)...) + f₂[2]subs(f₃[1],(x .=> y)...)subs(f₂[2],(x .=> z)...) + f₂[2]subs(f₃[2],(x .=> y)...)subs(f₂[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂[1]subs(f₂[1],(x .=> y)...)subs(f₃[1],(x .=> z)...) + f₂[1]subs(f₂[2],(x .=> y)...)subs(f₃[2],(x .=> z)...) + f₂[2]subs(f₂[1],(x .=> y)...)subs(f₃[2],(x .=> z)...) + f₂[2]subs(f₂[2],(x .=> y)...)subs(f₃[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 5
(f₂[1]subs(f₂[1],(x .=> y)...) + f₂[2]subs(f₂[2],(x .=> y)...))*subs(f₄, x=>z) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ
(f₂[1]subs(f₂[1],(x .=> z)...) + f₂[2]subs(f₂[2],(x .=> z)...))*subs(f₄, x=>y) # ρ₂₊₁ ⊗ ρₛ ⊗ ρ₂₊₁
f₄*(subs(f₂[1],(x .=> y)...)subs(f₂[1],(x .=> z)...) + subs(f₂[2],(x .=> y)...)subs(f₂[2],(x .=> z)...)) # ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₃[1]subs(f₃[1],(x .=> y)...)subs(f₂[1],(x .=> z)...) + f₃[1]subs(f₃[2],(x .=> y)...)subs(f₂[2],(x .=> z)...) + f₃[2]subs(f₃[1],(x .=> y)...)subs(f₂[2],(x .=> z)...) + f₃[2]subs(f₃[2],(x .=> y)...)subs(f₂[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₃[1]subs(f₂[1],(x .=> y)...)subs(f₃[1],(x .=> z)...) + f₃[1]subs(f₂[2],(x .=> y)...)subs(f₃[2],(x .=> z)...) + f₃[2]subs(f₂[1],(x .=> y)...)subs(f₃[2],(x .=> z)...) + f₃[2]subs(f₂[2],(x .=> y)...)subs(f₃[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂[1]subs(f₃[1],(x .=> y)...)subs(f₃[1],(x .=> z)...) + f₂[1]subs(f₃[2],(x .=> y)...)subs(f₃[2],(x .=> z)...) + f₂[2]subs(f₃[1],(x .=> y)...)subs(f₃[2],(x .=> z)...) + f₂[2]subs(f₃[2],(x .=> y)...)subs(f₃[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 6
(f₂[1]subs(f₃[1],(x .=> y)...) + f₂[2]subs(f₃[2],(x .=> y)...))*subs(f₄, x=>z) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ
(f₂[1]subs(f₃[1],(x .=> z)...) + f₂[2]subs(f₃[2],(x .=> z)...))*subs(f₄, x=>y) # ρ₂₊₁ ⊗ ρₛ ⊗ ρ₂₊₁
f₄*(subs(f₂[1],(x .=> y)...)subs(f₃[1],(x .=> z)...) + subs(f₂[2],(x .=> y)...)subs(f₃[2],(x .=> z)...)) # ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
(f₃[1]subs(f₂[1],(x .=> y)...) + f₃[2]subs(f₂[2],(x .=> y)...))*subs(f₄, x=>z) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ
(f₃[1]subs(f₂[1],(x .=> z)...) + f₃[2]subs(f₂[2],(x .=> z)...))*subs(f₄, x=>y) # ρ₂₊₁ ⊗ ρₛ ⊗ ρ₂₊₁
f₄*(subs(f₃[1],(x .=> y)...)subs(f₂[1],(x .=> z)...) + subs(f₃[2],(x .=> y)...)subs(f₂[2],(x .=> z)...)) # ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₃[1]subs(f₃[1],(x .=> y)...)subs(f₃[1],(x .=> z)...) + f₃[1]subs(f₃[2],(x .=> y)...)subs(f₃[2],(x .=> z)...) + f₃[2]subs(f₃[1],(x .=> y)...)subs(f₃[2],(x .=> z)...) + f₃[2]subs(f₃[2],(x .=> y)...)subs(f₃[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 7
(f₃[1]subs(f₃[1],(x .=> y)...) + f₃[2]subs(f₃[2],(x .=> y)...))*subs(f₄, x=>z) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ
(f₃[1]subs(f₃[1],(x .=> z)...) + f₃[2]subs(f₃[2],(x .=> z)...))*subs(f₄, x=>y) # ρ₂₊₁ ⊗ ρₛ ⊗ ρ₂₊₁
f₄*(subs(f₃[1],(x .=> y)...)subs(f₃[1],(x .=> z)...) + subs(f₃[2],(x .=> y)...)subs(f₃[2],(x .=> z)...)) # ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 9
f₄ * subs(f₄, x=>y) * subs(f₄, x=>z) # ρₛ ⊗ ρₛ ⊗ ρₛ
]

k = 1
for F in Fs
    @show k
    @test subs(F, x[1] => x[2], x[2] => x[1], y[1] => y[2], y[2] => y[1], z[1] => z[2], z[2] => z[1]) ≈ -F atol=1E-12
    @test subs(F,  ([x[2:n]; x[1]] .=> x)..., ([y[2:n]; y[1]] .=> y)..., ([z[2:n]; z[1]] .=> z)...) ≈ F atol=1E-12
    k += 1
end


######
# n = 4
######

n = 4
@polyvar x[1:n] monomial_order = Graded{DynamicPolynomials.Reverse{LexOrder}}
@polyvar y[1:n] monomial_order = Graded{DynamicPolynomials.Reverse{LexOrder}}
@polyvar z[1:n] monomial_order = Graded{DynamicPolynomials.Reverse{LexOrder}}


τ₁ = Permutation([[1,2], fill.(3:n,1)...])
c = Permutation([Vector(1:n)])

ρₚ₁ = polypermgen(n, 1)
ρₚ₂ = polypermgen(n, 2)
ρₚ₃ = polypermgen(n, 3)
@test all(subs(x, x[1] => x[2], x[2] => x[1]) .≈ ρₚ₁(τ₁) * x)
@test all(subs(x,  ([x[2:n]; x[1]] .=> x)...) .≈ ρₚ₁(c) * x)



f₄ = 1
f₃₊₁ = [(x[1]-x[2])/sqrt(2), (x[1]+x[2])/sqrt(6)-2x[3]/sqrt(6), (x[1]+x[2]+x[3])/sqrt(12)-sqrt(3)/2 * x[4]]
f₃₊₁₂ = [(x[1]^2-x[2]^2)/sqrt(2), (x[1]^2+x[2]^2)/sqrt(6)-2x[3]^2/sqrt(6), (x[1]^2+x[2]^2+x[3]^2)/sqrt(12)-sqrt(3)/2 * x[4]^2]
f₃₊₁₃ = [(x[1]^3-x[2]^3)/sqrt(2), (x[1]^3+x[2]^3)/sqrt(6)-2x[3]^3/sqrt(6), (x[1]^3+x[2]^3+x[3]^3)/sqrt(12)-sqrt(3)/2 * x[4]^3]
f₂₊₂ =  [(x[1]x[3]-x[1]x[4]-x[2]x[3]+x[2]x[4])/2, (x[1]x[2]+x[3]x[4])/sqrt(3) - (x[1]x[3]+x[1]x[4]+x[2]x[3]+x[2]x[4])/sqrt(12)]



@test all(f₃₊₁ .≈ blockdiagonalize(ρₚ₁)[2][:,1:3]'x)
@test all(f₂₊₂ .≈ blockdiagonalize(ρₚ₂)[2][:,1:2]'monomials(x, 2))

ρₜ = Representation(4)
ρ₃₊₁ = Representation(3,1)
ρ₂₊₂ = Representation(2,2)

@test all(subs(f₄, x[1] => x[2], x[2] => x[1]) .≈ ρₜ(τ₁) * f₄)
@test all(subs(f₄,  ([x[2:n]; x[1]] .=> x)...) .≈ ρₜ(c) * f₄)

@test all(subs(f₃₊₁, x[1] => x[2], x[2] => x[1]) .≈ ρ₃₊₁(τ₁) * f₃₊₁)
@test all(subs(f₃₊₁,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₃₊₁(c) * f₃₊₁)
@test all(subs(f₃₊₁₂, x[1] => x[2], x[2] => x[1]) .≈ ρ₃₊₁(τ₁) * f₃₊₁₂)
@test all(subs(f₃₊₁₂,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₃₊₁(c) * f₃₊₁₂)
@test all(subs(f₂₊₂, x[1] => x[2], x[2] => x[1]) .≈ ρ₂₊₂(τ₁) * f₂₊₂)
@test all(subs(f₂₊₂,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₂₊₂(c) * f₂₊₂)

# p = 0
f₄

# p = 1
@test length([
(x[1] + x[2] + x[3] + x[4])*f₄;
f₃₊₁
]) == 4

# p = 2
@test length([
(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*f₄;
(x[1]+x[2]+x[3]+x[4])^2*f₄;
(x[1] + x[2] + x[3] + x[4])*f₃₊₁;
f₃₊₁₂;
f₂₊₂
]) == size(ρₚ₂,1)

# p = 3
@test length([
(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*f₄;
(x[1]+x[2]+x[3]+x[4])*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*f₄;
(x[1]+x[2]+x[3]+x[4])^3*f₄;
(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^2*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])*f₃₊₁₂;
f₃₊₁₃
]