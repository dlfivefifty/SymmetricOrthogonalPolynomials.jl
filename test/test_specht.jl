using NumericalRepresentationTheory, DynamicPolynomials, Permutations

spechtpolynomial(yt::YoungMatrix, x) = prod(prod(prod(x[yt[ℓ,j]]-x[yt[k,j]] for ℓ=k+1:yt.columns[j]) for k = 1:yt.columns[j]-1; init=1) for j = 1:size(yt,2))
spechtpolynomial(yt::YoungTableau, x) = spechtpolynomial(YoungMatrix(yt), x)
spechtpolynomial(λ::Partition, x) = spechtpolynomial.(youngtableaux(λ), Ref(x))

x = randn(5)
y = randn(5)

@test spechtpolynomial(youngtableaux(Partition(2))[1], x) == 1
@test spechtpolynomial(youngtableaux(Partition(1,1))[1], x) == x[2]-x[1]
@test spechtpolynomial(youngtableaux(Partition(3))[1], x) == 1
@test spechtpolynomial(youngtableaux(Partition(2,1))[1], x) == x[2]-x[1]
@test spechtpolynomial(youngtableaux(Partition(2,1))[2], x) == x[3]-x[1]
@test spechtpolynomial(youngtableaux(Partition(1,1,1))[1], x) == (x[2]-x[1])*(x[3]-x[1])*(x[3]-x[2])
@test spechtpolynomial(youngtableaux(Partition(4))[1], x) == 1
@test spechtpolynomial(youngtableaux(Partition(3,1))[1], x) == x[2]-x[1]
@test spechtpolynomial(youngtableaux(Partition(3,1))[2], x) == x[3]-x[1]
@test spechtpolynomial(youngtableaux(Partition(3,1))[3], x) == x[4]-x[1]
@test spechtpolynomial(youngtableaux(Partition(2,2))[1], x) == (x[2]-x[1])*(x[4]-x[3])
@test spechtpolynomial(youngtableaux(Partition(2,2))[2], x) == (x[3]-x[1])*(x[4]-x[2])
@test spechtpolynomial(youngtableaux(Partition(2,1,1))[1], x) == (x[2]-x[1])*(x[3]-x[1])*(x[3]-x[2])
@test spechtpolynomial(youngtableaux(Partition(2,1,1))[2], x) == (x[2]-x[1])*(x[4]-x[1])*(x[4]-x[2])
@test spechtpolynomial(youngtableaux(Partition(2,1,1))[3], x) == (x[3]-x[1])*(x[4]-x[1])*(x[4]-x[3])
@test spechtpolynomial(youngtableaux(Partition(1,1,1,1))[1], x) ≈ (x[2]-x[1])*(x[3]-x[1])*(x[4]-x[1])*(x[3]-x[2])*(x[4]-x[2])*(x[4]-x[3])

λ = Partition(2,1)
n = Int(λ)
x = randn(n)
@test spechtpolynomial(λ, x[[2;1;3:n]]) ≈ [-1 0; -1 1] * spechtpolynomial(λ, x)
@test spechtpolynomial(λ, x[[1; 3; 2; 4:n]]) ≈ [0 1; 1 0] * spechtpolynomial(λ, x)

ρ = Representation(λ)

V = reshape(vec(nullspace([kron([-1 0; -1 1]', I(2)) - kron(I(2), ρ.generators[1]);
    kron([0 1; 1 0]', I(2)) - kron(I(2), ρ.generators[2])])), 2, 2)

@test V*[-1 0; -1 1] ≈ ρ.generators[1]*V
@test V*[0 1; 1 0] ≈ ρ.generators[2]*V

q = x -> V*spechtpolynomial(λ, x)
@test q(x[[2;1;3:n]]) ≈ ρ.generators[1] * q(x)
@test q(x[[1; 3; 2; 4:n]]) ≈ ρ.generators[2] * q(x)

λ = Partition(3,2)

n = Int(λ)
x = randn(n)
@test spechtpolynomial(λ,x) ≈ [
(x[2]-x[1])*(x[4]-x[3])
(x[2]-x[1])*(x[5]-x[3])
(x[3]-x[1])*(x[4]-x[2])
(x[3]-x[1])*(x[5]-x[2])
(x[4]-x[1])*(x[5]-x[2])
]

@polyvar x[1:n]
(x[4]-x[2])*(x[5]-x[1]) ≈  (x[4]-x[1])*(x[5]-x[2]) - (x[2]-x[1])*(x[5]-x[3]) + (x[2]-x[1])*(x[4]-x[3])
x = randn(n)
@test spechtpolynomial(λ, x[[2;1;3:n]]) ≈ [-1  0 0 0 0;
                                            0 -1 0 0 0;
                                           -1  0 1 0 0;
                                            0 -1 0 1 0;
                                            1 -1 0 0 1] * spechtpolynomial(λ, x)

@polyvar x[1:n]
@test (x[3]-x[1])*(x[4]-x[2]) ≈ (x[3]-x[1])*(x[4]-x[2])
(x[4]-x[1])*(x[5]-x[3]) - ((x[4]-x[1])*(x[5]-x[2])+ (x[2]-x[1])*(x[4]-x[3])-(x[3]-x[1])*(x[4]-x[2]))
x = randn(n)
@test spechtpolynomial(λ, x[[1; 3; 2; 4:n]]) ≈ [0 0 1 0 0;
                                                0 0 0 1 0;
                                                1 0 0 0 0;
                                                0 1 0 0 0
                                                1 0 -1 0 1] * spechtpolynomial(λ, x)


@test spechtpolynomial(λ, x[[1; 2; 4; 3; 5:n]]) ≈ [-1 0 0 0 0;
                                                -1 1 0 0 0;
                                                -1 0 1 0 0;
                                                0 0 0 0 1;
                                                0 0 0 1 0] * spechtpolynomial(λ, x)



@polyvar x[1:n]
x = randn(5)
@test (x[5]-x[1])*(x[4]-x[2]) ≈ ((x[4]-x[1])*(x[5]-x[2]) - (x[2]-x[1])*(x[5]-x[3]) + (x[2]-x[1])*(x[4]-x[3]))
@test spechtpolynomial(λ, x[[1:3; 5; 4]]) ≈ [0 1 0 0 0;
                                                1 0 0 0 0;
                                                0 0 0 1 0;
                                                0 0 1 0 0;
                                                1 -1 0 0 1] * spechtpolynomial(λ, x)


spechgens = ([-1  0 0 0 0;
            0 -1 0 0 0;
            -1  0 1 0 0;
            0 -1 0 1 0;
            1 -1 0 0 1],
            [0 0 1 0 0;
            0 0 0 1 0;
            1 0 0 0 0;
            0 1 0 0 0
            1 0 -1 0 1],
            [-1 0 0 0 0;
            -1 1 0 0 0;
            -1 0 1 0 0;
            0 0 0 0 1;
            0 0 0 1 0],
            [0 1 0 0 0;
            1 0 0 0 0;
            0 0 0 1 0;
            0 0 1 0 0;
            1 -1 0 0 1])


ρ = Representation(λ)

V = reshape(vec(nullspace([kron(spechgens[1]', I(5)) - kron(I(5), ρ.generators[1]);
    kron(spechgens[2]', I(5)) - kron(I(5), ρ.generators[2]);
    kron(spechgens[3]', I(5)) - kron(I(5), ρ.generators[3]);
    kron(spechgens[4]', I(5)) - kron(I(5), ρ.generators[4])
    ])), 5, 5)

for (σ,ρ) in zip(spechgens, ρ.generators)
    @test V*σ ≈ ρ*V
end

q = x -> V*spechtpolynomial(λ, x)
for k = 1:n-1
    @test q(x[[1:k-1; k+1; k; k+2:n]]) ≈ ρ.generators[k] * q(x)
end



###
# transpose
####

@polyvar x[1:n]
@test spechtpolynomial(λ',x) ≈ [
(x[2]-x[1])*(x[3]-x[1])*(x[3]-x[2])*(x[5]-x[4])
(x[2]-x[1])*(x[4]-x[1])*(x[4]-x[2])*(x[5]-x[3])
(x[2]-x[1])*(x[5]-x[1])*(x[5]-x[2])*(x[4]-x[3])
(x[3]-x[1])*(x[4]-x[1])*(x[4]-x[3])*(x[5]-x[2])
(x[3]-x[1])*(x[5]-x[1])*(x[5]-x[3])*(x[4]-x[2])
]

[coefficient.(basis_polys, m) for m in monomials_of_interest]

@polyvar x[1:n]
spechgenstrans =  ([-1 0 0 1 -1; 0 -1 0 -1 0; 0 0 -1 0 -1; 0 0 0 1 0; 0 0 0 0 1]',
                    [-1 0 0 0 0; 0 0 0 1 0; 0 0 0 0 1; 0 1 0 0 0; 0 0 1 0 0],
                    [0 1 0 0 0; 1 0 0 0 0; 0 0 -1 0 0; 0 0 0 -1 0; -1 1 -1 -1 1],
                    [-1 0 0 0 0; 0 0 1 0 0; 0 1 0 0 0; 0 0 0 0 1; 0 0 0 1 0])
@test (hcat(coefficients.(spechtpolynomial(λ', x[[2;1;3:n]]), Ref(monomials(x, 4)))...)\
    hcat(coefficients.(spechtpolynomial(λ', x), Ref(monomials(x, 4)))...))' ≈ spechgenstrans[1]
@test (hcat(coefficients.(spechtpolynomial(λ', x[[1; 3; 2; 4:n]]), Ref(monomials(x, 4)))...)\
    hcat(coefficients.(spechtpolynomial(λ', x), Ref(monomials(x, 4)))...))' ≈ spechgenstrans[2]
@test (hcat(coefficients.(spechtpolynomial(λ', x[[1; 2; 4; 3; 5:n]]), Ref(monomials(x, 4)))...)\
    hcat(coefficients.(spechtpolynomial(λ', x), Ref(monomials(x, 4)))...))' ≈ spechgenstrans[3]
@test (hcat(coefficients.(spechtpolynomial(λ', x[[1:3; 5; 4]]), Ref(monomials(x, 4)))...)\
    hcat(coefficients.(spechtpolynomial(λ', x), Ref(monomials(x, 4)))...))' ≈ spechgenstrans[4]
    

x = randn(n)
@test spechtpolynomial(λ', x[[2;1;3:n]]) ≈ spechgenstrans[1] * spechtpolynomial(λ', x)
@test spechtpolynomial(λ', x[[1; 3; 2; 4:n]]) ≈ spechgenstrans[2] * spechtpolynomial(λ', x)
@test spechtpolynomial(λ', x[[1; 2; 4; 3; 5:n]]) ≈ spechgenstrans[3] * spechtpolynomial(λ', x)
@test spechtpolynomial(λ', x[[1:3; 5; 4]]) ≈ spechgenstrans[4] * spechtpolynomial(λ', x)


ρt = Representation(λ')

Vt = reshape(vec(nullspace([kron(spechgenstrans[1]', I(5)) - kron(I(5), ρt.generators[1]);
    kron(spechgenstrans[2]', I(5)) - kron(I(5), ρt.generators[2]);
    kron(spechgenstrans[3]', I(5)) - kron(I(5), ρt.generators[3]);
    kron(spechgenstrans[4]', I(5)) - kron(I(5), ρt.generators[4])
    ])), 5, 5)

for (σ,ρ) in zip(spechgenstrans, ρt.generators)
    @test Vt*σ ≈ ρ*Vt
end

qt = x -> Vt*spechtpolynomial(λ', x)
for k = 1:n-1
    @test qt(x[[1:k-1; k+1; k; k+2:n]]) ≈ ρt.generators[k] * qt(x)
end


@test transpose.(YoungMatrix.(youngtableaux(λ))) == YoungMatrix.(youngtableaux(λ'))[end:-1:1]


####
# a fermion

y = randn(5)
f = (x,y) -> qt(x)'*Diagonal(sign.(YoungMatrix.(youngtableaux(λ))))[end:-1:1,:]*q(y)

@test f(x[[2;1; 3:n]], y[[2;1; 3:n]]) ≈ -f(x,y)
@test f(x[[2:n; 1]], y[[2:n; 1]]) ≈ f(x,y)

Vt'*Diagonal(sign.(YoungMatrix.(youngtableaux(λ))))[end:-1:1,:]*V


@test q(x[[1; 3; 2; 4:n]]) ≈ ρ.generators[2] * q(x)




yt = youngtableaux(Partition(3,1,1))[1]

@test spechtpolynomial(only(youngtableaux(Partition(1,1,1,1,1))), x) ≈ -spechtpolynomial(only(youngtableaux(Partition(1,1,1,1,1))), [x[2]; x[1]; x[3:end]])

yms =
sum(sign(ym)*spechtpolynomial(ym, x)spechtpolynomial(ym', y) for ym in yms)

n = length(x)

for λ in partitions(5)
    @show λ
    n = Int(λ)
    x = randn(n)
    y = randn(n)
    yms = YoungMatrix.(youngtableaux(λ))
    for k = 1:n-1
        τ₁ = [1:k-1; k+1; k; k+2:n]
        @test sum(sign(ym)*spechtpolynomial(ym, x[τ₁])spechtpolynomial(ym', y[τ₁]) for ym in yms) ≈ -sum(sign(ym)*spechtpolynomial(ym, x)spechtpolynomial(ym', y) for ym in yms)
    end
end




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


######
# orthogonal
######

gelfand(p, m) = sum(subs(p, x[k] => x[m+1], x[m+1] => x[k]) for k=1:m)
n = 2
@polyvar x[1:n]



# 2 = 2
p = 1
@test gelfand(p, 1) == p

# 2 = 1+1
p = x[2]-x[1]
@test gelfand(p, 1) == -p


n = 3
@polyvar x[1:n]

# 3 = 3
p = 1
@test gelfand(p, 1) == p
@test gelfand(p, 2) == 2p

# 2 = 2+1
p₁ = x[2]-x[1]
p₂ = x[3]-x[1] + x[3]-x[2]
@test gelfand(p₁, 1) == -p₁
@test gelfand(p₁, 2) == p₁
@test gelfand(p₂, 1) == p₂
@test gelfand(p₂, 2) == -p₂



n = 4
@polyvar x[1:n]

# 4 = 4
p = 1
@test gelfand(p, 1) == p
@test gelfand(p, 2) == 2p
@test gelfand(p, 3) == 3p

# 2 = 3+1
p₁ = x[2]-x[1] # [1 3 4; 2 0 0]
p₂ = x[3]-x[1] + x[3]-x[2] # [1 2 4; 3]
p₃ = x[4]-x[1] + x[4]-x[2] + x[4]-x[3] # [1 2 3; 4]
@test gelfand(p₁, 1) == -p₁
@test gelfand(p₁, 2) == p₁
@test gelfand(p₁, 3) == 2p₁
@test gelfand(p₂, 1) == p₂
@test gelfand(p₂, 2) == -p₂
@test gelfand(p₂, 3) == 2p₂
@test gelfand(p₃, 1) == p₃
@test gelfand(p₃, 2) == 2p₃
@test gelfand(p₃, 3) == -p₃




# 2 = 2+2
@polyvar c
p₁ = (x[2]-x[1]) * (x[4]-x[3]) # [1 3; 2 4]
p₂ = (x[4]-x[1] + x[4]-x[2]) + (x[4]-x[1] + x[4]-x[3]) + (x[4]-x[2] + x[4]-x[3])  # [1 2 4; 3]
@test gelfand(p₁, 1) == -p₁
@test gelfand(p₁, 2) == p₁
@test gelfand(p₁, 3) == 0
@test gelfand(p₂, 1) == p₂
@test gelfand(p₂, 2) == -p₂
@test gelfand(p₂, 3) == 0

