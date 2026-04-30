using DynamicPolynomials, NumericalRepresentationTheory, Permutations

function coeff_matrix(polys, monos)
    m = length(polys)
    n = length(monos)
    C = zeros(Float64, n, m)
    for (i, p) in enumerate(polys)
        # build a Dict: monomial => coefficient for fast lookup
        d = Dict(zip(monomials(p), coefficients(p)))
        for (j, mono) in enumerate(monos)
            C[j, i] = get(d, mono, 0.0)
        end
    end
    return C
end

𝐪 = λ -> blockdiagonalize(Representation(λ) ⊗ Representation(λ'))[2][:,1]

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

###
# n = 3 1D, All polys from Fake polys
###

fₜ = 1 # trivial
f₂₊₁ = [(x[1]-x[2])/sqrt(2), (x[1]+x[2])/sqrt(6) - sqrt(2/3) * x[3]]
f₂₊₁₂ = [(x[1]^2-x[2]^2)/sqrt(2), (x[1]^2+x[2]^2)/sqrt(6) - sqrt(2/3) * x[3]^2]
fₛ = (x[1]-x[2])*(x[1]-x[3])*(x[2]-x[3])


ρₜ = Representation(3)
ρ₂₊₁ = Representation(2,1)
ρₛ = Representation(1,1,1)

τ₁ = Permutation([[1,2], [3]])
c = Permutation([[1,2,3]])


@test all(subs(fₜ, x[1] => x[2], x[2] => x[1]) .≈ ρₜ(τ₁) * fₜ)
@test all(subs(fₜ,  ([x[2:n]; x[1]] .=> x)...) .≈ ρₜ(c) * fₜ)

@test all(subs(f₂₊₁, x[1] => x[2], x[2] => x[1]) .≈ ρ₂₊₁(τ₁) * f₂₊₁)
@test all(subs(f₂₊₁,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₂₊₁(c) * f₂₊₁)

@test all(subs(f₂₊₁₂, x[1] => x[2], x[2] => x[1]) .≈ ρ₂₊₁(τ₁) * f₂₊₁₂)
@test all(subs(f₂₊₁₂,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₂₊₁(c) * f₂₊₁₂)

@test subs(fₛ , x[1] => x[2], x[2] => x[1]) ≈ only(ρₛ(τ₁)) * fₛ
@test subs(fₛ ,  ([x[2:n]; x[1]] .=> x)...) .≈ only(ρₛ(c)) * fₛ



# p = 0
fₜ

# p = 1
@test rank(coeff_matrix(
    [
        (x[1]+x[2]+x[3])*fₜ;
        f₂₊₁], monomials(x, 1))
        ) == 3

# p = 2
@test length([(x[1]^2+x[2]^2+x[3]^2)*fₜ;
    (x[1]+x[2]+x[3])^2*fₜ;
    (x[1]+x[2]+x[3])*f₂₊₁;
    f₂₊₁₂;
    ]) == 6

# p = 3
@test length([(x[1]^3+x[2]^3+x[3]^3)*fₜ;
    (x[1]+x[2]+x[3])*(x[1]^2+x[2]^2+x[3]^2)*fₜ;
    (x[1]+x[2]+x[3])^3*fₜ;
    (x[1]^2+x[2]^2+x[3]^2)*f₂₊₁;
    (x[1]+x[2]+x[3])^2*f₂₊₁;
    (x[1]+x[2]+x[3])*f₂₊₁₂;
    fₛ ;
    ]) == 10



####
# 2D
####

@test multiplicities(ρₛ ⊗ ρₜ)[Partition(1,1,1)] == multiplicities(ρₜ ⊗ ρₛ)[Partition(1,1,1)] == 1
@test multiplicities(ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1
λ,Q = blockdiagonalize(ρ₂₊₁ ⊗ ρ₂₊₁)
@test Q[:,1] ≈ [0,1/sqrt(2),-1/sqrt(2),0]

# p = 2
F₃ = f₂₊₁[2]subs(f₂₊₁[1],(x .=> y)...) - f₂₊₁[1]subs(f₂₊₁[2],(x .=> y)...)

# p = 3
F₁ = fₛ  # ρₛ ⊗ ρₜ
F₂ = subs(fₛ ,(x .=> y)...) # ρₜ ⊗ ρₛ
F₄ = f₂₊₁[2]subs(f₂₊₁₂[1],(x .=> y)...) - f₂₊₁[1]subs(f₂₊₁₂[2],(x .=> y)...)
F₅ = f₂₊₁₂[2]subs(f₂₊₁[1],(x .=> y)...) - f₂₊₁₂[1]subs(f₂₊₁[2],(x .=> y)...)
# + invariant * F₃
# F₃ * (x[1]+x[2]+x[3])
# F₃ * (y[1]+y[2]+y[3])

# p = 4
F₆ = f₂₊₁₂[2]subs(f₂₊₁₂[1],(x .=> y)...) - f₂₊₁₂[1]subs(f₂₊₁₂[2],(x .=> y)...)


for F in (F₁,F₂,F₃,F₄,F₅,F₆)
    @test subs(F, x[1] => x[2], x[2] => x[1], y[1] => y[2], y[2] => y[1]) ≈ -F
    @test subs(F,  ([x[2:n]; x[1]] .=> x)..., ([y[2:n]; y[1]] .=> y)...) ≈ F
end

####
# 3D
####

@test multiplicities(ρₜ ⊗ ρₜ ⊗ ρₛ)[Partition(1,1,1)] == 1
@test multiplicities(ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1
@test multiplicities(ρₛ ⊗ ρₛ ⊗ ρₛ)[Partition(1,1,1)] == 1
@test multiplicities(ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1
@test multiplicities(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1
@test !haskey(multiplicities(ρₜ ⊗ ρₜ ⊗ ρₜ), Partition(1,1,1))
@test !haskey(multiplicities(ρₜ ⊗ ρₜ ⊗ ρ₂₊₁), Partition(1,1,1))
@test !haskey(multiplicities(ρₜ ⊗ ρₛ ⊗ ρₛ), Partition(1,1,1))
@test !haskey(multiplicities(ρₜ ⊗ ρₛ ⊗ ρ₂₊₁), Partition(1,1,1))
@test !haskey(multiplicities(ρₛ ⊗ ρₛ ⊗ ρ₂₊₁), Partition(1,1,1))

@test blockdiagonalize(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ)[2][:,1] ≈ [1,  0,    
                                                    0,  1]/sqrt(2)



@test blockdiagonalize(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[2][:,1] ≈ [-1,0,    0,1,    
                                                       0,1,     1,0]/2


Fs = [
# p = 2
F₃ # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₜ
subs(F₃,(y .=> z)...) # ρ₂₊₁ ⊗ ρₜ ⊗ ρ₂₊₁
subs(F₃,(x .=> y)..., (y .=> z)...) # ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 3
fₛ # ρₛ ⊗ ρₜ ⊗ ρₜ
subs(fₛ,(x .=> y)...) # ρₜ ⊗ ρₛ ⊗ ρₜ
subs(fₛ,(x .=> z)...) # ρₜ ⊗ ρₜ ⊗ ρₛ
F₄ # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₜ
subs(F₄,(y .=> z)...) # ρ₂₊₁ ⊗ ρₜ ⊗ ρ₂₊₁
subs(F₄,(x .=> y)..., (y .=> z)...) # ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
F₅ # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₜ
subs(F₅,(y .=> z)...) # ρ₂₊₁ ⊗ ρₜ ⊗ ρ₂₊₁
subs(F₅,(x .=> y)..., (y .=> z)...) # ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂₊₁[1]subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) + f₂₊₁[1]subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁


# p = 4
F₆ # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₜ
subs(F₆,(y .=> z)...) # ρ₂₊₁ ⊗ ρₜ ⊗ ρ₂₊₁
subs(F₆,(x .=> y)..., (y .=> z)...) # ρₜ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂₊₁₂[1]subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) + f₂₊₁₂[1]subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂₊₁[1]subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) + f₂₊₁[1]subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂₊₁[1]subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) + f₂₊₁[1]subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 5
(f₂₊₁[1]subs(f₂₊₁[1],(x .=> y)...) + f₂₊₁[2]subs(f₂₊₁[2],(x .=> y)...))*subs(fₛ , x=>z) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ
(f₂₊₁[1]subs(f₂₊₁[1],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁[2],(x .=> z)...))*subs(fₛ , x=>y) # ρ₂₊₁ ⊗ ρₛ ⊗ ρ₂₊₁
fₛ *(subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) + subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...)) # ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂₊₁₂[1]subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) + f₂₊₁₂[1]subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂₊₁₂[1]subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) + f₂₊₁₂[1]subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂₊₁[1]subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) + f₂₊₁[1]subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 6
(f₂₊₁[1]subs(f₂₊₁₂[1],(x .=> y)...) + f₂₊₁[2]subs(f₂₊₁₂[2],(x .=> y)...))*subs(fₛ , x=>z) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ
(f₂₊₁[1]subs(f₂₊₁₂[1],(x .=> z)...) + f₂₊₁[2]subs(f₂₊₁₂[2],(x .=> z)...))*subs(fₛ , x=>y) # ρ₂₊₁ ⊗ ρₛ ⊗ ρ₂₊₁
fₛ *(subs(f₂₊₁[1],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) + subs(f₂₊₁[2],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...)) # ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
(f₂₊₁₂[1]subs(f₂₊₁[1],(x .=> y)...) + f₂₊₁₂[2]subs(f₂₊₁[2],(x .=> y)...))*subs(fₛ , x=>z) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ
(f₂₊₁₂[1]subs(f₂₊₁[1],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁[2],(x .=> z)...))*subs(fₛ , x=>y) # ρ₂₊₁ ⊗ ρₛ ⊗ ρ₂₊₁
fₛ *(subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁[1],(x .=> z)...) + subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁[2],(x .=> z)...)) # ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁
-f₂₊₁₂[1]subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) + f₂₊₁₂[1]subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 7
(f₂₊₁₂[1]subs(f₂₊₁₂[1],(x .=> y)...) + f₂₊₁₂[2]subs(f₂₊₁₂[2],(x .=> y)...))*subs(fₛ , x=>z) # ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ
(f₂₊₁₂[1]subs(f₂₊₁₂[1],(x .=> z)...) + f₂₊₁₂[2]subs(f₂₊₁₂[2],(x .=> z)...))*subs(fₛ , x=>y) # ρ₂₊₁ ⊗ ρₛ ⊗ ρ₂₊₁
fₛ *(subs(f₂₊₁₂[1],(x .=> y)...)subs(f₂₊₁₂[1],(x .=> z)...) + subs(f₂₊₁₂[2],(x .=> y)...)subs(f₂₊₁₂[2],(x .=> z)...)) # ρₛ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁

# p = 9
fₛ  * subs(fₛ , x=>y) * subs(fₛ , x=>z) # ρₛ ⊗ ρₛ ⊗ ρₛ
]


@test rank(coeff_matrix(Fs, vcat((monomials([x;y;z], k) for k=0:9)...))) == 36


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
ρₚ₄ = polypermgen(n, 4)
ρₚ₅ = polypermgen(n, 5)
@test all(subs(x, x[1] => x[2], x[2] => x[1]) .≈ ρₚ₁(τ₁) * x)
@test all(subs(x,  ([x[2:n]; x[1]] .=> x)...) .≈ ρₚ₁(c) * x)

ρₜ = Representation(4)
ρ₃₊₁ = Representation(3,1)
ρ₂₊₂ = Representation(2,2)
ρ₂₊₁₊₁ = Representation(2,1,1)
ρₛ = Representation(1,1,1,1)


fₜ = 1
f₃₊₁ = [(x[1]-x[2])/sqrt(2), (x[1]+x[2])/sqrt(6)-2x[3]/sqrt(6), (x[1]+x[2]+x[3])/sqrt(12)-sqrt(3)/2 * x[4]]
f₃₊₁₂ = [(x[1]^2-x[2]^2)/sqrt(2), (x[1]^2+x[2]^2)/sqrt(6)-2x[3]^2/sqrt(6), (x[1]^2+x[2]^2+x[3]^2)/sqrt(12)-sqrt(3)/2 * x[4]^2]
f₃₊₁₃ = [(x[1]^3-x[2]^3)/sqrt(2), (x[1]^3+x[2]^3)/sqrt(6)-2x[3]^3/sqrt(6), (x[1]^3+x[2]^3+x[3]^3)/sqrt(12)-sqrt(3)/2 * x[4]^3]
f₂₊₂ =  -[(x[1]x[3]-x[1]x[4]-x[2]x[3]+x[2]x[4])/2, (x[1]x[2]+x[3]x[4])/sqrt(3) - (x[1]x[3]+x[1]x[4]+x[2]x[3]+x[2]x[4])/sqrt(12)]
f₂₊₂₂ =  [(x[1]^2*x[3]^2-x[1]^2*x[4]^2-x[2]^2*x[3]^2+x[2]^2*x[4]^2)/2, (x[1]^2*x[2]^2+x[3]^2*x[4]^2)/sqrt(3) - (x[1]^2*x[3]^2+x[1]^2*x[4]^2+x[2]^2*x[3]^2+x[2]^2*x[4]^2)/sqrt(12)]
f₂₊₁₊₁ = [(x[1]^2*x[2]-x[1]^2*x[3]-x[1]x[2]^2+x[1]x[3]^2+x[2]^2*x[3]-x[2]x[3]^2)/sqrt(6),
          (2x[1]^2*x[2]+x[1]^2*x[3]-3x[1]^2*x[4]-2x[1]x[2]^2-x[1]x[3]^2+3x[1]x[4]^2-x[2]^2*x[3]+3x[2]^2*x[4]+x[2]x[3]^2-3x[2]x[4]^2)/sqrt(48),
          (x[1]^2*x[3]-x[1]^2*x[4]-x[1]x[3]^2+x[1]x[4]^2+x[2]^2*x[3]-x[2]^2*x[4]-x[2]x[3]^2+x[2]x[4]^2+2x[3]^2*x[4]-2x[3]x[4]^2)/4]
f₂₊₁₊₁₂ = [x[1]^2*x[2]x[4]-x[1]^2*x[3]x[4]-x[1]x[2]^2*x[4]+x[1]x[3]^2*x[4]+x[2]^2*x[3]x[4]-x[2]x[3]^2*x[4],
            (3x[1]^2*x[2]x[3]-x[1]^2*x[2]x[4]-2x[1]^2*x[3]x[4]-3x[1]x[2]^2*x[3]+x[1]x[2]^2*x[4]-x[1]x[3]^2*x[4]+3x[1]x[3]x[4]^2+2x[2]^2*x[3]x[4]+x[2]x[3]^2*x[4]-3x[2]x[3]x[4]^2)/sqrt(8),
            (x[1]^2*x[2]x[3]-x[1]^2*x[2]x[4]+x[1]x[2]^2*x[3]-x[1]x[2]^2*x[4]-2x[1]x[2]x[3]^2+2x[1]x[2]x[4]^2+x[1]x[3]^2*x[4]-x[1]x[3]x[4]^2+x[2]x[3]^2*x[4]-x[2]x[3]x[4]^2)/sqrt(8/3)
            ]
f₂₊₁₊₁₃ = [(x[1]^2*x[2]^3-x[1]^2*x[3]^3-x[1]^3*x[2]^2+x[1]^3*x[3]^2+x[2]^2*x[3]^3-x[2]^3*x[3]^2)/sqrt(6),
          (2x[1]^2*x[2]^3+x[1]^2*x[3]^3-3x[1]^2*x[4]^3-2x[1]^3*x[2]^2-x[1]^3*x[3]^2+3x[1]^3*x[4]^2-x[2]^2*x[3]^3+3x[2]^2*x[4]^3+x[2]^3*x[3]^2-3x[2]^3*x[4]^2)/sqrt(48),
          (x[1]^2*x[3]^3-x[1]^2*x[4]^3-x[1]^3*x[3]^2+x[1]^3*x[4]^2+x[2]^2*x[3]^3-x[2]^2*x[4]^3-x[2]^3*x[3]^2+x[2]^3*x[4]^2+2x[3]^2*x[4]^3-2x[3]^3*x[4]^2)/4]
fₛ = (x[1]-x[2])*(x[1]-x[3])*(x[1]-x[4])*(x[2]-x[3])*(x[2]-x[4])*(x[3]-x[4])

@test all(f₃₊₁ .≈ blockdiagonalize(ρₚ₁)[2][:,1:3]'x)
@test all(f₂₊₂ .≈ blockdiagonalize(ρₚ₂)[2][:,1:2]'monomials(x, 2))
@test all(f₂₊₁₊₁ .≈ blockdiagonalize(ρₚ₃)[2][:,1:3]'monomials(x, 3))


A = (blockdiagonalize(ρₚ₄)[2][:,1:3] - blockdiagonalize(ρₚ₄)[2][:,1:3][2]/coeff_matrix((x[1]+x[2]+x[3]+x[4])*f₂₊₁₊₁, monomials(x,4))[2] * coeff_matrix((x[1]+x[2]+x[3]+x[4])*f₂₊₁₊₁, monomials(x,4)))
B = A/A[7]
@test B ≈ coeff_matrix(f₂₊₁₊₁₂, monomials(x,4))


@test all(subs(fₜ, x[1] => x[2], x[2] => x[1]) .≈ ρₜ(τ₁) * fₜ)
@test all(subs(fₜ,  ([x[2:n]; x[1]] .=> x)...) .≈ ρₜ(c) * fₜ)

@test all(subs(f₃₊₁, x[1] => x[2], x[2] => x[1]) .≈ ρ₃₊₁(τ₁) * f₃₊₁)
@test all(subs(f₃₊₁,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₃₊₁(c) * f₃₊₁)
@test all(subs(f₃₊₁₂, x[1] => x[2], x[2] => x[1]) .≈ ρ₃₊₁(τ₁) * f₃₊₁₂)
@test all(subs(f₃₊₁₂,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₃₊₁(c) * f₃₊₁₂)
@test all(subs(f₂₊₂, x[1] => x[2], x[2] => x[1]) .≈ ρ₂₊₂(τ₁) * f₂₊₂)
@test all(subs(f₂₊₂,  ([x[2:n]; x[1]] .=> x)...) .≈ ρ₂₊₂(c) * f₂₊₂)
@test all(subs(f₂₊₁₊₁, x[1] => x[2], x[2] => x[1]) .≈ ρ₂₊₁₊₁(τ₁) * f₂₊₁₊₁)
@test all(broadcast((a,b) -> ≈(a,b;atol=1E-12), subs(f₂₊₁₊₁, ([x[2:n]; x[1]] .=> x)...), ρ₂₊₁₊₁(c) * f₂₊₁₊₁))
@test all(subs(f₂₊₁₊₁₂, x[1] => x[2], x[2] => x[1]) .≈ ρ₂₊₁₊₁(τ₁) * f₂₊₁₊₁₂)
@test all(broadcast((a,b) -> ≈(a,b;atol=1E-12), subs(f₂₊₁₊₁₂, ([x[2:n]; x[1]] .=> x)...), ρ₂₊₁₊₁(c) * f₂₊₁₊₁₂))
@test all(subs(f₂₊₁₊₁₃, x[1] => x[2], x[2] => x[1]) .≈ ρ₂₊₁₊₁(τ₁) * f₂₊₁₊₁₃)
@test all(broadcast((a,b) -> ≈(a,b;atol=1E-12), subs(f₂₊₁₊₁₃, ([x[2:n]; x[1]] .=> x)...), ρ₂₊₁₊₁(c) * f₂₊₁₊₁₃))


# p = 0
fₜ

# p = 1
@test length([
(x[1] + x[2] + x[3] + x[4])*fₜ;
f₃₊₁
]) == 4

# p = 2
@test rank(coeff_matrix([
(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*fₜ;
(x[1]+x[2]+x[3]+x[4])^2*fₜ;
(x[1] + x[2] + x[3] + x[4])*f₃₊₁;
f₃₊₁₂;
f₂₊₂
], monomials(x,2))) == size(ρₚ₂,1)

# p = 3
@test rank(coeff_matrix([
(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*fₜ;
(x[1]+x[2]+x[3]+x[4])*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*fₜ;
(x[1]+x[2]+x[3]+x[4])^3*fₜ;
(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^2*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])*f₃₊₁₂;
f₃₊₁₃;
(x[1]+x[2]+x[3]+x[4])*f₂₊₂;
f₂₊₁₊₁
], monomials(x,3))) == size(ρₚ₃,1)

# p = 4
@test rank(coeff_matrix([
(x[1]^4 + x[2]^4 + x[3]^4 + x[4]^4)*fₜ;
(x[1]+x[2]+x[3]+x[4])*(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*fₜ;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*fₜ;
(x[1]+x[2]+x[3]+x[4])^4*fₜ;
(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)^2*fₜ;
(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^3*f₃₊₁;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])^2*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])*f₃₊₁₃;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])^2*f₂₊₂;
f₂₊₂₂;
(x[1]+x[2]+x[3]+x[4])*f₂₊₁₊₁;
f₂₊₁₊₁₂
], monomials(x,4))) == size(ρₚ₄,1)

# p = 5
@test rank(coeff_matrix([
(x[1]^5 + x[2]^5 + x[3]^5 + x[4]^5)*fₜ;
(x[1]+x[2]+x[3]+x[4])*(x[1]^4 + x[2]^4 + x[3]^4 + x[4]^4)*fₜ;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*fₜ;
(x[1]+x[2]+x[3]+x[4])^3*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*fₜ;
(x[1]+x[2]+x[3]+x[4])^5*fₜ;
(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*fₜ;
(x[1]^4 + x[2]^4 + x[3]^4 + x[4]^4)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])*(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^4*f₃₊₁;
(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)^2*f₃₊₁;
(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])^3*f₃₊₁₂;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₃₊₁₃;
(x[1]+x[2]+x[3]+x[4])^2*f₃₊₁₃;
(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])^3*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])*f₂₊₂₂;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₁₊₁;
(x[1]+x[2]+x[3]+x[4])^2*f₂₊₁₊₁;
(x[1]+x[2]+x[3]+x[4])*f₂₊₁₊₁₂;
f₂₊₁₊₁₃
], monomials(x,5))) == size(ρₚ₅,1)


# p = 6
@test rank(coeff_matrix([
(x[1]^6 + x[2]^6 + x[3]^6 + x[4]^6)*fₜ;
(x[1]+x[2]+x[3]+x[4])*(x[1]^5 + x[2]^5 + x[3]^5 + x[4]^5)*fₜ;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^4 + x[2]^4 + x[3]^4 + x[4]^4)*fₜ;
(x[1]+x[2]+x[3]+x[4])^3*(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*fₜ;
(x[1]+x[2]+x[3]+x[4])^4*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*fₜ;
(x[1]+x[2]+x[3]+x[4])^6*fₜ;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*(x[1]^4+x[2]^4+x[3]^4+x[4]^4)*fₜ;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^3*fₜ;
(x[1]^3+x[2]^3+x[3]^3+x[4]^3)^2*fₜ;
(x[1]^5 + x[2]^5 + x[3]^5 + x[4]^5)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])*(x[1]^4 + x[2]^4 + x[3]^4 + x[4]^4)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^3*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^5*f₃₊₁;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₃₊₁;
(x[1]^4+x[2]^4+x[3]^4+x[4]^4)*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])^4*f₃₊₁₂;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^2*f₃₊₁₂;
(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₃₊₁₃;
(x[1]+x[2]+x[3]+x[4])*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₃₊₁₃;
(x[1]+x[2]+x[3]+x[4])^3*f₃₊₁₃;
(x[1]^4+x[2]^4+x[3]^4+x[4]^4)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])^4*f₂₊₂;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^2*f₂₊₂;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₂₂;
(x[1]+x[2]+x[3]+x[4])^2*f₂₊₂₂;
(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₂₊₁₊₁;
(x[1]+x[2]+x[3]+x[4])*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₁₊₁;
(x[1]+x[2]+x[3]+x[4])^3*f₂₊₁₊₁;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₁₊₁₂;
(x[1]+x[2]+x[3]+x[4])^2*f₂₊₁₊₁₂;
(x[1]+x[2]+x[3]+x[4])*f₂₊₁₊₁₃;
fₛ
], monomials(x,6))) == size(polypermgen(n,6),1)


@test rank(coeff_matrix([
(x[1]^7 + x[2]^7 + x[3]^7 + x[4]^7)*fₜ;
(x[1]+x[2]+x[3]+x[4])*(x[1]^6 + x[2]^6 + x[3]^6 + x[4]^6)*fₜ;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^5 + x[2]^5 + x[3]^5 + x[4]^5)*fₜ;
(x[1]+x[2]+x[3]+x[4])^3*(x[1]^4 + x[2]^4 + x[3]^4 + x[4]^4)*fₜ;
(x[1]+x[2]+x[3]+x[4])^4*(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*fₜ;
(x[1]+x[2]+x[3]+x[4])^5*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*fₜ;
(x[1]+x[2]+x[3]+x[4])^7*fₜ;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*(x[1]^5+x[2]^5+x[3]^5+x[4]^5)*fₜ;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^2*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*fₜ;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^3*(x[1]+x[2]+x[3]+x[4])*fₜ;
(x[1]^3+x[2]^3+x[3]^3+x[4]^3)^2*(x[1]+x[2]+x[3]+x[4])*fₜ;
(x[1]^6 + x[2]^6 + x[3]^6 + x[4]^6)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])*(x[1]^5 + x[2]^5 + x[3]^5 + x[4]^5)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^4 + x[2]^4 + x[3]^4 + x[4]^4)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^3*(x[1]^3 + x[2]^3 + x[3]^3 + x[4]^3)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^4*(x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2)*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])^6*f₃₊₁;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*(x[1]^4+x[2]^4+x[3]^4+x[4]^4)*f₃₊₁;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^3*f₃₊₁;
(x[1]^3+x[2]^3+x[3]^3+x[4]^3)^2*f₃₊₁;
(x[1]+x[2]+x[3]+x[4])*(x[1]^4+x[2]^4+x[3]^4+x[4]^4)*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])^3*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₃₊₁₂;
(x[1]+x[2]+x[3]+x[4])^5*f₃₊₁₂;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^2*(x[1]+x[2]+x[3]+x[4])*f₃₊₁₂;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₃₊₁₂;
(x[1]^4+x[2]^4+x[3]^4+x[4]^4)*f₃₊₁₃;
(x[1]+x[2]+x[3]+x[4])*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₃₊₁₃;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₃₊₁₃;
(x[1]+x[2]+x[3]+x[4])^4*f₃₊₁₃;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^2*f₃₊₁₃;
(x[1]^5+x[2]^5+x[3]^5+x[4]^5)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])*(x[1]^4+x[2]^4+x[3]^4+x[4]^4)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])^3*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₂;
(x[1]+x[2]+x[3]+x[4])^5*f₂₊₂;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^2*(x[1]+x[2]+x[3]+x[4])*f₂₊₂;
# (x[1]^2+x[2]^2+x[3]^2+x[4]^2)*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₂₊₂;
(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₂₊₂₂;
(x[1]+x[2]+x[3]+x[4])*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₂₂;
(x[1]+x[2]+x[3]+x[4])^3*f₂₊₂₂;
(x[1]^4+x[2]^4+x[3]^4+x[4]^4)*f₂₊₁₊₁;
(x[1]+x[2]+x[3]+x[4])*(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₂₊₁₊₁;
(x[1]+x[2]+x[3]+x[4])^2*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₁₊₁;
(x[1]+x[2]+x[3]+x[4])^4*f₂₊₁₊₁;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)^2*f₂₊₁₊₁;
(x[1]^3+x[2]^3+x[3]^3+x[4]^3)*f₂₊₁₊₁₂;
(x[1]+x[2]+x[3]+x[4])*(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₁₊₁₂;
(x[1]+x[2]+x[3]+x[4])^3*f₂₊₁₊₁₂;
(x[1]^2+x[2]^2+x[3]^2+x[4]^2)*f₂₊₁₊₁₃;
(x[1]+x[2]+x[3]+x[4])^2*f₂₊₁₊₁₃;
(x[1]+x[2]+x[3]+x[4])*fₛ
], monomials(x,7))) == size(polypermgen(n,7),1)


###
# n = 4, 2D
###

@test multiplicities(ρₜ ⊗ ρₛ)[Partition(1,1,1,1)] == 1
@test multiplicities(ρ₃₊₁ ⊗ ρ₂₊₁₊₁)[Partition(1,1,1,1)] == 1
@test multiplicities(ρ₂₊₂ ⊗ ρ₂₊₂)[Partition(1,1,1,1)] == 1

@test !haskey(multiplicities(ρ₃₊₁ ⊗ ρ₃₊₁),Partition(1,1,1,1))
@test !haskey(multiplicities(ρ₂₊₁₊₁ ⊗ ρ₂₊₁₊₁),Partition(1,1,1,1))
@test !haskey(multiplicities(ρ₂₊₁₊₁ ⊗ ρ₂₊₂),Partition(1,1,1,1))
@test !haskey(multiplicities(ρ₃₊₁ ⊗ ρₛ),Partition(1,1,1,1))
@test !haskey(multiplicities(ρ₂₊₁₊₁ ⊗ ρₛ),Partition(1,1,1,1))
@test !haskey(multiplicities(ρₛ ⊗ ρ₂₊₂),Partition(1,1,1,1))

@test blockdiagonalize(ρ₃₊₁ ⊗ ρ₂₊₁₊₁)[2][:,1] ≈ [0,0,-1,0,1,0,-1,0,0]/sqrt(3)
@test blockdiagonalize(ρ₂₊₂ ⊗ ρ₂₊₂)[2][:,1] ≈ [0,1,-1,0]/sqrt(2)

x2y = f -> subs(f,(x .=> y)...)

Fs = [
    # p = 4
    f₂₊₂[2]x2y(f₂₊₂[1])-f₂₊₂[1]x2y(f₂₊₂[2])
    -f₂₊₁₊₁[3]x2y(f₃₊₁[1])+f₂₊₁₊₁[2]x2y(f₃₊₁[2])-f₂₊₁₊₁[1]x2y(f₃₊₁[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₃₊₁[3]x2y(f₂₊₁₊₁[1])+f₃₊₁[2]x2y(f₂₊₁₊₁[2])-f₃₊₁[1]x2y(f₂₊₁₊₁[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    # p = 5
    -f₂₊₁₊₁[3]x2y(f₃₊₁₂[1])+f₂₊₁₊₁[2]x2y(f₃₊₁₂[2])-f₂₊₁₊₁[1]x2y(f₃₊₁₂[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₃₊₁₂[3]x2y(f₂₊₁₊₁[1])+f₃₊₁₂[2]x2y(f₂₊₁₊₁[2])-f₃₊₁₂[1]x2y(f₂₊₁₊₁[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₂₊₁₊₁₂[3]x2y(f₃₊₁[1])+f₂₊₁₊₁₂[2]x2y(f₃₊₁[2])-f₂₊₁₊₁₂[1]x2y(f₃₊₁[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₃₊₁[3]x2y(f₂₊₁₊₁₂[1])+f₃₊₁[2]x2y(f₂₊₁₊₁₂[2])-f₃₊₁[1]x2y(f₂₊₁₊₁₂[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    # p = 6
    -f₂₊₁₊₁₂[3]x2y(f₃₊₁₂[1])+f₂₊₁₊₁₂[2]x2y(f₃₊₁₂[2])-f₂₊₁₊₁₂[1]x2y(f₃₊₁₂[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₃₊₁₂[3]x2y(f₂₊₁₊₁₂[1])+f₃₊₁₂[2]x2y(f₂₊₁₊₁₂[2])-f₃₊₁₂[1]x2y(f₂₊₁₊₁₂[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    f₂₊₂₂[2]x2y(f₂₊₂[1])-f₂₊₂₂[1]x2y(f₂₊₂[2])
    f₂₊₂[2]x2y(f₂₊₂₂[1])-f₂₊₂[1]x2y(f₂₊₂₂[2])
    -f₂₊₁₊₁[3]x2y(f₃₊₁₃[1])+f₂₊₁₊₁[2]x2y(f₃₊₁₃[2])-f₂₊₁₊₁[1]x2y(f₃₊₁₃[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₃₊₁₃[3]x2y(f₂₊₁₊₁[1])+f₃₊₁₃[2]x2y(f₂₊₁₊₁[2])-f₃₊₁₃[1]x2y(f₂₊₁₊₁[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₂₊₁₊₁₃[3]x2y(f₃₊₁[1])+f₂₊₁₊₁₃[2]x2y(f₃₊₁[2])-f₂₊₁₊₁₃[1]x2y(f₃₊₁[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₃₊₁[3]x2y(f₂₊₁₊₁₃[1])+f₃₊₁[2]x2y(f₂₊₁₊₁₃[2])-f₃₊₁[1]x2y(f₂₊₁₊₁₃[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    fₛ # ρₜ ⊗ ρₛ
    x2y(fₛ)# ρₜ ⊗ ρₛ
    # p = 7
    -f₂₊₁₊₁₂[3]x2y(f₃₊₁₃[1])+f₂₊₁₊₁₂[2]x2y(f₃₊₁₃[2])-f₂₊₁₊₁₂[1]x2y(f₃₊₁₃[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₃₊₁₃[3]x2y(f₂₊₁₊₁₂[1])+f₃₊₁₃[2]x2y(f₂₊₁₊₁₂[2])-f₃₊₁₃[1]x2y(f₂₊₁₊₁₂[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₂₊₁₊₁₃[3]x2y(f₃₊₁₂[1])+f₂₊₁₊₁₃[2]x2y(f₃₊₁₂[2])-f₂₊₁₊₁₃[1]x2y(f₃₊₁₂[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₃₊₁₂[3]x2y(f₂₊₁₊₁₃[1])+f₃₊₁₂[2]x2y(f₂₊₁₊₁₃[2])-f₃₊₁₂[1]x2y(f₂₊₁₊₁₃[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    # p = 8
    f₂₊₂₂[2]x2y(f₂₊₂₂[1])-f₂₊₂₂[1]x2y(f₂₊₂₂[2])
    -f₂₊₁₊₁₃[3]x2y(f₃₊₁₃[1])+f₂₊₁₊₁₃[2]x2y(f₃₊₁₃[2])-f₂₊₁₊₁₃[1]x2y(f₃₊₁₃[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
    -f₃₊₁₃[3]x2y(f₂₊₁₊₁₃[1])+f₃₊₁₃[2]x2y(f₂₊₁₊₁₃[2])-f₃₊₁₃[1]x2y(f₂₊₁₊₁₃[3]) # ρ₃₊₁ ⊗ ρ₂₊₁₊₁
]



for F in Fs
    @test subs(F, x[1] => x[2], x[2] => x[1], y[1] => y[2], y[2] => y[1]) ≈ -F atol=1E-13
    @test subs(F,  ([x[2:n]; x[1]] .=> x)..., ([y[2:n]; y[1]] .=> y)...) ≈ sign(c)F atol=1E-13
end


###
# 3D
###

@test multiplicities(ρₜ ⊗ ρₜ ⊗ ρₛ)[Partition(1,1,1,1)] == 1
@test multiplicities(ρₜ ⊗ ρ₃₊₁ ⊗ ρ₂₊₁₊₁)[Partition(1,1,1,1)] == 1
@test multiplicities(ρₜ ⊗ ρ₂₊₂ ⊗ ρ₂₊₂)[Partition(1,1,1,1)] == 1
@test multiplicities(ρ₂₊₂ ⊗ ρ₂₊₂ ⊗ ρ₂₊₂)[Partition(1,1,1,1)] == 1
@test multiplicities(ρ₃₊₁ ⊗ ρ₃₊₁ ⊗ ρ₂₊₂)[Partition(1,1,1,1)] == 1
@test multiplicities(ρ₃₊₁ ⊗ ρ₃₊₁ ⊗ ρ₃₊₁)[Partition(1,1,1,1)] == 1
@test multiplicities(ρ₃₊₁ ⊗ ρ₃₊₁ ⊗ ρ₂₊₁₊₁)[Partition(1,1,1,1)] == 1
@test multiplicities(ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₁₊₁)[Partition(1,1,1,1)] == 1
@test multiplicities(ρ₂₊₁₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₁₊₁)[Partition(1,1,1,1)] == 1
@test multiplicities(ρ₃₊₁ ⊗ ρ₂₊₂ ⊗ ρ₂₊₁₊₁)[Partition(1,1,1,1)] == 1
@test !haskey(multiplicities(ρ₃₊₁ ⊗ ρ₂₊₂ ⊗ ρ₂₊₂), Partition(1,1,1,1))

τ₁ = Permutation([[1,2], [3],[4]])
c = Permutation([[1,2,3,4]])
@test (ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₂)(τ₁) ≈ kron(ρ₃₊₁(τ₁), ρ₂₊₁₊₁(τ₁),ρ₂₊₂(τ₁)) ≈  kron(kron(ρ₃₊₁(τ₁), ρ₂₊₁₊₁(τ₁)),ρ₂₊₂(τ₁))
@test (ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₂)(c) ≈ kron(ρ₃₊₁(c), ρ₂₊₁₊₁(c),ρ₂₊₂(c)) ≈ kron(kron(ρ₃₊₁(c), ρ₂₊₁₊₁(c)),ρ₂₊₂(c))

@test blockdiagonalize(ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₂)[2][:,1] ≈ [sqrt(2),0,  1,0,        0,-1,
                                                        0,sqrt(2),  0,-1,       -1,0,
                                                        0,0,        0,-sqrt(2), sqrt(2),0]/sqrt(12)
@test blockdiagonalize(ρ₃₊₁ ⊗ ρ₃₊₁ ⊗ ρ₂₊₂)[2][:,1] ≈ [1,0,0,-1,0,sqrt(2),0,-1,-1,0,-sqrt(2),0,0,sqrt(2),-sqrt(2),0,0,0]/sqrt(12)
@test blockdiagonalize(ρ₂₊₂ ⊗ ρ₂₊₂ ⊗ ρ₂₊₂)[2][:,1] ≈ [-1,0,0,1,0,1,1,0]/2
@test blockdiagonalize(ρ₃₊₁ ⊗ ρ₃₊₁ ⊗ ρ₃₊₁)[2][:,1] ≈ [0,0,0,0,0,-1,0,1,0,0,0,1,0,0,0,-1,0,0,0,-1,0,1,0,0,0,0,0]/sqrt(6)
blockdiagonalize(ρ₂₊₁₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₁₊₁)[2][:,1]
blockdiagonalize(ρ₃₊₁ ⊗ ρ₃₊₁ ⊗ ρ₂₊₁₊₁)[2][:,1]


### higher n in 2D

dat = [[2],
[1 , 4, 1],
[3, 4, 10, 4, 3],
[3, 8, 15, 20, 28, 20, 15, 8, 3],
[1 , 10, 16, 40, 58, 84, 93, 116, 93, 84, 58, 40, 16, 10, 1],
[4, 15, 40, 75, 140, 216, 312, 403, 488, 539, 576, 539, 488, 403, 312, 216, 140, 75, 40, 15, 4],
[6, 20, 65, 126, 264, 444, 733, 1060, 1518, 1958, 2481, 2888, 3292, 3488, 3634, 3488, 3292,  2888, 2481, 1958, 1518, 1060, 733, 444, 264, 126, 65, 20, 6],
[4, 25, 68, 179, 380, 726, 1280, 2140, 3294, 4881, 6846, 9221, 11890, 14829, 17730, 20566, 23004, 24939, 26128, 26620, 26128, 24939, 23004, 20566, 17730, 14829, 11890, 9221, 6846, 4881, 3294, 2140, 1280, 726, 380, 179, 68, 25, 4],
[ 1 , 18, 58, 180, 415, 916, 1741, 3214, 5417, 8846, 13589, 20256, 28792, 39898, 53101, 69008,  86687,  106468,  126820,  147908,  167666,  186230,  201405,  213498,  220548,  223440,  220548,  213498,  201405,  186230,  167666,  147908,  126820,  106468,  86687,  69008,  53101,  39898,  28792,  20256,  13589,  8846,  5417,  3214,  1741,  916,  415,  180,  58,  18,  1]
]

k = 1;
datk = k -> map(d -> k ≤ length(d) ? d[k] : NaN, dat)
plot([datk(1) datk(2) datk(3) datk(4) datk(5)]; legend=:topleft, label = (1:3)')

plot([datk(1) datk(2) datk(3) datk(4) datk(5) datk(6)]; legend=:topleft, label = (1:6)', xscale=:log10, yscale=:log10)
plot!((1:9).^3; label=nothing, linestyle=:dash)
annotate!([(3,100,"O(n^3)")])



####
# Slater determinants
####


# n = 3
n = 3
@polyvar x[1:n]
@polyvar y[1:n]

# p = 2
det([ones(n) x y])

# p = 3
det([ones(n) x x.^2])
det([ones(n) x x.*y])
det([ones(n) x y.^2])
det([ones(n) y x.^2])
det([ones(n) y x.*y])
det([ones(n) y y.^2])

# p = 4
C = coeff_matrix([
det([ones(n) x x.^3])
det([ones(n) x (x.^2).*y])
det([ones(n) x x.*y.^2])
det([ones(n) x y.^3])
det([ones(n) y x.^3])
det([ones(n) y (x.^2).*y])
det([ones(n) y x.*y.^2])
det([ones(n) y y.^3])
det([ones(n) x.^2 x.*y])
det([ones(n) x.^2 y.^2])
det([ones(n) y.^2 x.*y])
det([x y x.^2])
det([x y x.*y])
det([x y y.^2])
], monomials([x;y], 4))
@test rank(C) == 14

# degree of p = 2invariant polys:
@test 6 == 2multiplicities(polypermgen(3,2) ⊗ polypermgen(3,0))[Partition(n)] +# 2*2
multiplicities(polypermgen(3,1) ⊗ polypermgen(3,1))[Partition(n)]# 2
# total # of p = 4 invariant polys:
5 + 2*4 + 1

C₂ = [
F₃*(x[1]^2+x[2]^2+x[3]^2)
F₃*(y[1]^2+y[2]^2+y[3]^2)
F₃*(x[1]+x[2]+x[3])^2
F₃*(y[1]+y[2]+y[3])^2
F₃*(x[1]+x[2]+x[3])*(y[1]+y[2]+y[3])
F₁*(x[1]+x[2]+x[3])
F₁*(y[1]+y[2]+y[3])
F₂*(x[1]+x[2]+x[3])
F₂*(y[1]+y[2]+y[3])
F₄*(x[1]+x[2]+x[3])
F₄*(y[1]+y[2]+y[3])
F₅*(x[1]+x[2]+x[3])
F₅*(y[1]+y[2]+y[3])
F₆
]

for F in C₂
    @test subs(F, x[1] => x[2], x[2] => x[1], y[1] => y[2], y[2] => y[1]) ≈ -F
    @test subs(F,  ([x[2:n]; x[1]] .=> x)..., ([y[2:n]; y[1]] .=> y)...) ≈ F atol=1E-12
end

nullspace(coeff_matrix(C₂, monomials([x;y],4)))


c = coeff_matrix([F₆],monomials([x;y], 4))
@test C*(C\c) ≈ c

# n = 4
n = 4


# p = 4
det([ones(4) x y x.^2])
det([ones(4) x y y.^2])
det([ones(4) x y x.*y])


# p = 5
Ss = [
det([ones(4) x x.^2 x.*y])
det([ones(4) x x.^2 y.^2])
det([ones(4) x x.*y y.^2])
det([ones(4) y x.^2 y.^2])
det([ones(4) y x.^2 x.*y])
det([ones(4) y x.*y y.^2])
det([ones(4) x y x.^3])
det([ones(4) x y (x.^2).*y])
det([ones(4) x y x.*y.^2])
det([ones(4) x y y.^3])
]

#2 + 3

rank(coeff_matrix(Ss, monomials([x;y],5)))

# p = 6
det([ones(4) x.^2 x.*y y.^2])
det([x y x.^2 y.^2])
det([x y x.^2 x.*y])





#




q = blockdiagonalize(Representation(3,2,1) ⊗ Representation(3,2,1))[2][:,1]

blockdiagonalize(Representation((Representation(3,2,1) ⊗ Representation(3,2,1)).generators[1:end-1]))[2][:,1:3]


n = 8
count(!iszero, round.(blockdiagonalize( Representation(n-2,1,1) ⊗ Representation(3, fill(1,n-3)...))[2][:,1]; digits=6))




𝐪(Partition(2,1))

F_λ = function(λ)
    Ys = youngtableaux(λ)
    𝐪_λ = 𝐪(λ)
    s = sign.(𝐪_λ[findall(!iszero, round.(𝐪_λ; digits=10))])

end


filter(!iszero, vec(YoungMatrix(youngtableaux(Partition(3,2,1))[1])))

λ = Partition(3,2,1)
YT = YoungMatrix.(youngtableaux(λ))
qq = map(yt -> sign(Permutation(filter(!iszero, vec(yt)))), YT)
@test qq == -sign.(filter(!iszero, round.(𝐪(λ);digits=10)))

for k = 1:12
    @show k
    for λ in partitions(k)
        ρ  = Representation(λ)
        ρ′ = Representation(λ')
        YT = YoungMatrix.(youngtableaux(λ))
        qq = map(yt -> sign(Permutation(filter(!iszero, vec(yt)))), YT)
        𝐪𝐪 = vec(Diagonal(qq)[end:-1:1,:])

        for g in (ρ ⊗ ρ′).generators
            @test g * 𝐪𝐪 ≈ -𝐪𝐪
        end
    end
end



####
# proof of 𝐪𝐪

λ = Partition(2,1)
ρ  = Representation(λ)
ρ′ = Representation(λ')
YT = YoungMatrix.(youngtableaux(λ))
qq = map(yt -> sign(Permutation(filter(!iszero, vec(yt)))), YT)
Σ = Diagonal(qq)[end:-1:1,:]

for g in PermGen(3)
    @test ρ(g) * Σ * ρ′(g)' ≈ sign(g) * Σ
end


@test ρ.generators[2] * Σ * ρ′.generators[2] ≈ -Σ

Representation(λ).generators




###
# build up 3D
####

# n = 2
multiplicities(Representation(1,1) ⊗ Representation(2) ⊗ Representation(2))
multiplicities(Representation(1,1) ⊗ Representation(1,1) ⊗ Representation(1,1))

# n = 3

r = 2; Q = [1/r sqrt(1-1/r^2); sqrt(1-1/r^2) -1/r]
X = reshape(blockdiagonalize(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₜ)[2][:,1], 1,2,2)

@test ρ₂₊₁.generators[2] ≈ Q
τ₁,τ₂ = (ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₜ).generators
@test τ₁ ≈ kron(Diagonal([-1,1]), Diagonal([-1,1]), [1;;])
@test X[1,1,1] == X[1,2,2] == 0 # since τ₁*X Hits X[1,1,1] by 1^3 and X[1,2,2] by (-1)^2*1
@test τ₂ ≈ kron(Q, Q, [1;;])
@test τ₂ * vec([0 1; -1 0]) ≈ kron(Q,Q) * vec([0 1; -1 0]) ≈ -vec([0 1; -1 0])
@test X[1,1,2] ≈ - X[1,2,1]

# therefore X = c*[0 1; - 1 0]


X = reshape(blockdiagonalize(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ)[2][:,1], 1, 2, 2)
τ₁,τ₂ = (ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρₛ).generators
@test τ₁ ≈ kron(Diagonal([-1,1]), Diagonal([-1,1]), [-1;;])
@test X[1,1,2] == X[1,2,1] == 0 # since τ₁*X Hits X[1,1,2] by (-1)*(-1)*1 and X[1,2,1] by (-1)*(-1)*1
@test τ₂ ≈ kron(Q, Q, [-1;;])
@test kron(Q,Q) * vec([1 0; 0 1]) ≈ vec([1 0; 0 1])
@test τ₂ * vec([1 0; 0 1]) ≈ -vec([1 0; 0 1])
@test X[1,1,1] ≈ X[1,2,2]
# therefore X = c*[1 0; 0 1]

X = reshape(blockdiagonalize(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[2][:,1], 2, 2, 2)
τ₁,τ₂ = (ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁).generators
@test τ₁ ≈ kron(Diagonal([-1,1]), Diagonal([-1,1]), Diagonal([-1,1]))
@test X[1,1,2] == X[1,2,1] == X[2,1,1] == X[2,2,2] == 0
@test τ₂ ≈ kron(Q, Q, Q)
# we need to find the eigvector that is zero in the right entries
# we can do this via a filter:
P = I(size(X,1)^3)[[1,4,6,7],:]
Q̃ = P * kron(Q, Q, Q) * P'
@test Q̃ * [-1,1,1,1] ≈ -[-1,1,1,1]
# thus we have
@test -X[1,1,1] ≈ X[1,2,2] ≈ X[2,1,2] ≈ X[2,2,1]
# therefore X = c*[[-1 0; 0 1]; [0 1; 1 0]]
@test X ≈ [[-1 0; 0 1];;; [0 1; 1 0]]/2

# note we can think of it as the following SYT that are interacting:
# 1 3   ⊗   1   3  ⊗    1   3
# 2         2           2
# 1 3   ⊗   1   2  ⊗    1   2
# 2         3           3
# 1 2   ⊗   1   3  ⊗    1   2
# 3         2           3
# 1 2   ⊗   1   2  ⊗    1   3
# 3         3           2


# because when we swap 2 and 3 all of these are inter-connected

X = reshape(blockdiagonalize(ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₂)[2][:,1], 2, 3, 3)
τ₁,τ₂,τ₃ = (ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₂).generators
@test τ₁ ≈ kron(Diagonal([-1,1,1]), Diagonal([-1,-1,1]), Diagonal([-1,1]))
@test X[2,1,1] == X[2,2,1] == X[1,3,1] == X[1,1,2] == X[1,2,2] == X[2,3,2] == X[1,1,3]  == X[1,2,3] == X[2,3,3] == 0


# when we drop the last blocks we have copies of
# 2+1 ⊗ 1+1+1 ⊗ 2+1 
# from entries
# 1:2 ⊗ 1:1 ⊗ 1:2
# hence

@test sqrt(3) * vec(X[1:2,1:1,1:2]) ≈ blockdiagonalize(Representation(2,1) ⊗ Representation(1,1,1) ⊗ Representation(2,1))[2][:,1]

# 2+1 ⊗ 2+1 ⊗ 2+1 
# from entries
# 1:2 ⊗ 2:3 ⊗ 1:2
# hence
@test -sqrt(3) * vec(X[1:2,2:3,1:2]) ≈ blockdiagonalize(Representation(2,1) ⊗ Representation(2,1) ⊗ Representation(2,1))[2][:,1]

# 3 ⊗ 1+1+1 ⊗ 2+1 
# from entries
# 3:3 ⊗ 1:1 ⊗ 1:2
# but this has no sign representation hence
@test all(iszero, X[1:2,1:1,3:3])

# 3 ⊗ 2+1 ⊗ 2+1 
# from entries
# 3:3 ⊗ 2:3 ⊗ 1:2
@test -sqrt(3) * vec(X[1:2,2:3,3:3]) ≈ blockdiagonalize(Representation(3) ⊗ Representation(2,1) ⊗ Representation(2,1))[2][:,1]

# this has established a zero pattern
# we now consider τ_3 which  permutes 3 and 4
# We know X[1,1,1] (linear index 1) has SYT
# 1 3 4     ⊗   1 4     ⊗       1 3
# 2             2               2 4
#               3
# interacts with X[1,2,1] (linear index 3) with SYT
# 1 3 4     ⊗   1 3     ⊗       1 3
# 2             2               2 4
#               4
# according to the following:


r = 3; Q = [1/r sqrt(1-1/r^2); sqrt(1-1/r^2) -1/r]
@test τ₃[[1,3],[1,3]] ≈ -Q ≈ kron([1;;],Q,[-1;;])
@test τ₃[[1,3],[1,3]] * [sqrt(2),1] ≈ -[sqrt(2),1]
@test τ₃[[1,3],[1,3]]*X[[1,3]] ≈ - X[[1,3]]
@test X[[1,3]] ≈ 1/sqrt(12) * [sqrt(2),1]

# hence we know the ratio:
@test X[1,1,1]/sqrt(2) ≈ X[1,2,1]

# or in linear indexing
@test X[1]/sqrt(2) ≈ X[3]

# that is, we can relate the blocks 
# 2+1 ⊗ 1+1+1 ⊗ 2+1     (X[1:2,1:1,1:2])
# to
# 2+1 ⊗ 2+1 ⊗ 2+1       (X[1:2,2:3,1:2]))
# we now need to relate that last group (3 ⊗ 2+1 ⊗ 2+1)
# whose indices range X[1:2,2:3,3:3].
# We know X[1,2,3] (linear index 15) has SYT
# 1 2 3     ⊗   1 3     ⊗       1 3
# 4             2               2 4
#               4
# but this will be zero since when we drop 4 then 3 we are
# left with 2 ⊗ 1+1 ⊗ 1+1 which has no sign representation. Similar
#  with X[1,1,3] (linear index 13) with SYT
# 1 2 3     ⊗   1 4     ⊗       1 3
# 4             2               2 4
#               3
# and  with X[1,1,2] (linear index 7) with SYT
# 1 2 4     ⊗   1 4     ⊗       1 3
# 3             2               2 4
#               3
# and  with X[1,2,2] (linear index 9) with SYT
# 1 2 4     ⊗   1 3     ⊗       1 3
# 3             2               2 4
#               4
# Thus consider the next SYT in this family
# X[1,3,3] (linear indexing 17)
# 1 2 3     ⊗   1 2     ⊗       1 3
# 4             3               2 4
#               4
# which only interacts with X[1,3,2] (linear indexing 11)
# 1 2 4     ⊗   1 2     ⊗       1 3
# 3             3               2 4
#               4
# according to the following:


r = 3; Q = [1/r sqrt(1-1/r^2); sqrt(1-1/r^2) -1/r]

@test Representation(3,1).generators[3][2:3,2:3] ≈ Q

@test τ₃[[11,17],[11,17]] ≈ Q ≈ kron(Q,[-1;;],[-1;;])
@test τ₃[[11,17],[11,17]] * [1,-sqrt(2)] ≈ -[1,-sqrt(2)]
@test τ₃[[11,17],[11,17]]*X[[11,17]] ≈ -X[[11,17]]
@test X[[11,17]] ≈ -1/sqrt(12) * [1,-sqrt(2)]
# hence we know the ratio:
@test X[1,3,2]*sqrt(2) ≈ -X[1,3,3]


# thus we have arrived an algorithm for building up X (up to a constant):

X̃ = similar(X); X̃ .= NaN
X̃[1:2,1:1,1:2] = blockdiagonalize(Representation(2,1) ⊗ Representation(1,1,1) ⊗ Representation(2,1))[2][:,1]
X̃[1:2,2:3,1:2] = blockdiagonalize(Representation(2,1) ⊗ Representation(2,1) ⊗ Representation(2,1))[2][:,1]
X̃[1:2,1:1,3:3] .= 0
X̃[1:2,2:3,3:3] = blockdiagonalize(Representation(3) ⊗ Representation(2,1) ⊗ Representation(2,1))[2][:,1]
@test !any(isnan, X̃) # we have non-zero entries
@test (ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₂).generators[1] * vec(X̃) ≈ -vec(X̃)
@test (ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₂).generators[2] * vec(X̃) ≈ -vec(X̃)
# we haven't got the normalisation in yet! We need to change the constants
@test !((ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₂).generators[3] * vec(X̃) ≈ -vec(X̃))

X̃[1:2,2:3,1:2] *= X̃[1,1,1]/(sqrt(2)X̃[1,2,1])
X̃[1:2,2:3,3:3] *= -X̃[1,3,2]*sqrt(2)/X̃[1,3,3]
@test (ρ₃₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₂).generators[3] * vec(X̃) ≈ -vec(X̃)





#### tree


𝐪 = Dict()

λ₃ = Partition(3)
λ₂₊₁ = Partition(2,1)
λ₁₊₁₊₁ = Partition(1,1,1)


ρ₃ = Representation(3)
ρ₂₊₁ = Representation(2,1)
ρ₁₊₁₊₁ = Representation(1,1,1)
Q₂₊₁ = ρ₂₊₁.generators[2]

@test multiplicities(ρ₁₊₁₊₁ ⊗ ρ₁₊₁₊₁ ⊗ ρ₁₊₁₊₁)[Partition(1,1,1)] == 1

𝐪[(λ₁₊₁₊₁ , λ₁₊₁₊₁ , λ₁₊₁₊₁)] = [1]
let 𝐯 = 𝐪[(λ₁₊₁₊₁ , λ₁₊₁₊₁ , λ₁₊₁₊₁)]
    for g in (ρ₁₊₁₊₁ ⊗ ρ₁₊₁₊₁ ⊗ ρ₁₊₁₊₁).generators
        @test g*𝐯 ≈ -𝐯
    end
end

@test !haskey(multiplicities(ρ₁₊₁₊₁ ⊗ ρ₁₊₁₊₁ ⊗ ρ₂₊₁), Partition(1,1,1))
@test !haskey(multiplicities(ρ₁₊₁₊₁ ⊗ ρ₂₊₁ ⊗ ρ₁₊₁₊₁), Partition(1,1,1))

@test multiplicities(ρ₁₊₁₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1
@test eigen(Matrix(kron(-I(1), Q₂₊₁, Q₂₊₁))[[1,4],[1,4]]).vectors[:,1] ≈ [1,1]/sqrt(2)

𝐪[(λ₁₊₁₊₁, λ₂₊₁, λ₂₊₁)] = [1,0,0,1]
let 𝐯 = 𝐪[(λ₁₊₁₊₁ , λ₂₊₁ , λ₂₊₁)]
    for g in (ρ₁₊₁₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁).generators
        @test g*𝐯 ≈ -𝐯
    end
end

@test !haskey(multiplicities(ρ₁₊₁₊₁ ⊗ ρ₂₊₁ ⊗ ρ₃), Partition(1,1,1))
@test !haskey(multiplicities(ρ₁₊₁₊₁ ⊗ ρ₃ ⊗ ρ₁₊₁₊₁), Partition(1,1,1))
@test !haskey(multiplicities(ρ₁₊₁₊₁ ⊗ ρ₃ ⊗ ρ₂₊₁), Partition(1,1,1))

@test multiplicities(ρ₁₊₁₊₁ ⊗ ρ₃ ⊗ ρ₃)[Partition(1,1,1)] == 1

𝐪[(λ₁₊₁₊₁ , λ₃ , λ₃)] = [1]
let 𝐯 = 𝐪[(λ₁₊₁₊₁ , λ₃ , λ₃)]
    for g in (ρ₁₊₁₊₁ ⊗ ρ₃ ⊗ ρ₃).generators
        @test g*𝐯 ≈ -𝐯
    end
end

@test !haskey(multiplicities(ρ₂₊₁ ⊗ ρ₁₊₁₊₁ ⊗ ρ₁₊₁₊₁), Partition(1,1,1))
@test multiplicities(ρ₂₊₁ ⊗ ρ₁₊₁₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1 # see above
@test multiplicities(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₁₊₁₊₁)[Partition(1,1,1)] == 1 # see above

@test multiplicities(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1

𝐪[(λ₂₊₁ , λ₂₊₁ , λ₂₊₁)] = [1,0,0,-1,0,-1,-1,0]
let 𝐯 = 𝐪[(λ₂₊₁ , λ₂₊₁ , λ₂₊₁)]
    for g in (ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁).generators
        @test g*𝐯 ≈ -𝐯
    end
end

@test multiplicities(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₃)[Partition(1,1,1)] == 1

𝐪[(λ₂₊₁ , λ₂₊₁ , λ₃)] =  [0,1,-1,0]
let 𝐯 =𝐪[(λ₂₊₁ , λ₂₊₁ , λ₃)]
    for g in (ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₃).generators
        @test g*𝐯 ≈ -𝐯
    end
end

@test !haskey(multiplicities(ρ₂₊₁ ⊗ ρ₃ ⊗ ρ₁₊₁₊₁), Partition(1,1,1))
@test multiplicities(ρ₂₊₁ ⊗ ρ₃ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1 # see above
@test !haskey(multiplicities(ρ₂₊₁ ⊗ ρ₃ ⊗ ρ₃), Partition(1,1,1))

@test !haskey(multiplicities(ρ₃ ⊗ ρ₁₊₁₊₁ ⊗ ρ₁₊₁₊₁), Partition(1,1,1))
@test !haskey(multiplicities(ρ₃ ⊗ ρ₁₊₁₊₁ ⊗ ρ₂₊₁), Partition(1,1,1))
@test multiplicities(ρ₃ ⊗ ρ₁₊₁₊₁ ⊗ ρ₃)[Partition(1,1,1)] == 1 # see above
@test !haskey(multiplicities(ρ₃ ⊗ ρ₂₊₁ ⊗ ρ₁₊₁₊₁), Partition(1,1,1))
@test multiplicities(ρ₃ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[Partition(1,1,1)] == 1 # see above
@test !haskey(multiplicities(ρ₃ ⊗ ρ₂₊₁ ⊗ ρ₃), Partition(1,1,1))
@test multiplicities(ρ₃ ⊗ ρ₃ ⊗ ρ₁₊₁₊₁)[Partition(1,1,1)] == 1 # see above
@test !haskey(multiplicities(ρ₃ ⊗ ρ₃ ⊗ ρ₂₊₁), Partition(1,1,1))
@test !haskey(multiplicities(ρ₃ ⊗ ρ₃ ⊗ ρ₃), Partition(1,1,1))

# configuration

@test eigvals(Matrix(kron(Q₂₊₁,Q₂₊₁,Q₂₊₁)[[1,4,6,7],[1,4,6,7]])) ≈ [-1,1,1,1]
@test eigen(Matrix(kron(Q₂₊₁,Q₂₊₁,Q₂₊₁)[[1,4,6,7],[1,4,6,7]])).vectors[:,1] ≈ [1,-1,-1,-1]/2

𝐪₂₊₁ = blockdiagonalize(ρ₂₊₁ ⊗ ρ₂₊₁ ⊗ ρ₂₊₁)[2][:,1]

Q = ρ₂₊₁.generators[2]



# n = 4


λ₄ = Partition(4)
λ₃₊₁ = Partition(3,1)
λ₂₊₂ = Partition(2,2)
λ₂₊₁₊₁ = Partition(2,1,1)
λ₁₊₁₊₁₊₁ = Partition(1,1,1,1)

ρ₄ = Representation(4)
ρ₃₊₁ = Representation(3,1)
ρ₂₊₂ = Representation(2,2)
ρ₂₊₁₊₁ = Representation(2,1,1)
ρ₁₊₁₊₁₊₁ = Representation(1,1,1,1)

@test multiplicities(ρ₁₊₁₊₁₊₁ ⊗ ρ₁₊₁₊₁₊₁ ⊗ ρ₁₊₁₊₁₊₁)[λ₁₊₁₊₁₊₁] == 1
let 𝐯 = [1]
    for g in (ρ₁₊₁₊₁₊₁ ⊗ ρ₁₊₁₊₁₊₁ ⊗ ρ₁₊₁₊₁₊₁).generators
        @test g*𝐯 ≈ -𝐯
    end
end

@test !haskey(multiplicities(ρ₁₊₁₊₁₊₁ ⊗ ρ₁₊₁₊₁₊₁ ⊗ ρ₂₊₁₊₁),λ₁₊₁₊₁₊₁)
@test !haskey(multiplicities(ρ₁₊₁₊₁₊₁ ⊗ ρ₁₊₁₊₁₊₁ ⊗ ρ₂₊₂),λ₁₊₁₊₁₊₁)
@test !haskey(multiplicities(ρ₁₊₁₊₁₊₁ ⊗ ρ₁₊₁₊₁₊₁ ⊗ ρ₃₊₁),λ₁₊₁₊₁₊₁)
@test !haskey(multiplicities(ρ₁₊₁₊₁₊₁ ⊗ ρ₁₊₁₊₁₊₁ ⊗ ρ₄),λ₁₊₁₊₁₊₁)


@test !haskey(multiplicities(ρ₁₊₁₊₁₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₁₊₁₊₁₊₁),λ₁₊₁₊₁₊₁)

@test multiplicities(ρ₁₊₁₊₁₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₁₊₁)[λ₁₊₁₊₁₊₁] == 1
𝐪[(λ₁₊₁₊₁₊₁ , λ₂₊₁₊₁ , λ₂₊₁₊₁)] =  zeros(size(ρ₁₊₁₊₁₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₁₊₁,1))
𝐪[(λ₁₊₁₊₁₊₁ , λ₂₊₁₊₁ , λ₂₊₁₊₁)][1:1] = 𝐪[(λ₁₊₁₊₁, λ₁₊₁₊₁, λ₁₊₁₊₁)]
𝐪[(λ₁₊₁₊₁₊₁ , λ₂₊₁₊₁ , λ₂₊₁₊₁)][[5,6,8,9]] = 𝐪[(λ₁₊₁₊₁, λ₂₊₁, λ₂₊₁)]

let 𝐯 = 𝐪[(λ₁₊₁₊₁₊₁ , λ₂₊₁₊₁ , λ₂₊₁₊₁)]
    for g in (ρ₁₊₁₊₁₊₁ ⊗ ρ₂₊₁₊₁ ⊗ ρ₂₊₁₊₁).generators[1:1]
        @test g*𝐯 ≈ -𝐯
    end
end


# not all multiplicities are 1 or 0
@test multiplicities(Representation(3,2,1) ⊗ Representation(4,2) ⊗ Representation(3,2,1))[Partition(fill(1,6)...)] == 3