using NumericalRepresentationTheory, DynamicPolynomials, Permutations

gelfand(p, m) = sum(subs(p, x[k] => x[m+1], x[m+1] => x[k]) for k=1:m)

####
# classic specht
####

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
p¹⁺¹ = [x[2]-x[1]]
@test subs(p¹⁺¹, x[2]=>x[1], x[1]=>x[2]) == -p¹⁺¹


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

q₂₊₁ =Diagonal([sqrt(3)/2, x[3]-x[1]/2-x[2]/2]) * [p¹⁺¹..., p₂]
@test all(q₂₊₁ .≈ [sqrt(3)/2 * (x[2]-x[1]);
        x[3]-x[1]/2-x[2]/2])
@test all(q₂₊₁ .≈ V\p₂₊₁)
@test all(subs(q₂₊₁, x[2]=>x[1], x[1]=>x[2]) .≈
            Diagonal([sqrt(3)/2, x[3]-x[1]/2-x[2]/2]) * subs([p¹⁺¹..., p₂], x[2]=>x[1], x[1]=>x[2]) .≈
            Diagonal([sqrt(3)/2, x[3]-x[1]/2-x[2]/2]) * blockdiag(ρ₁₊₁(τ₁), ρ₂(τ₁)) * [p¹⁺¹..., p₂] .≈
            ρ₂₊₁(τ₁)*q₂₊₁)
@test all(subs(q₂₊₁, x[2]=>x[3], x[3]=>x[2]) .≈ ρ₂₊₁(τ₂)*q₂₊₁)




¹⁺¹⁺¹ = p¹⁺¹ * (x[1]-x[3]) * (x[2]-x[3])
@test subs(¹⁺¹⁺¹, x[2]=>x[1], x[1]=>x[2]) == ρ₁₊₁₊₁(τ₁)*¹⁺¹⁺¹
@test subs(¹⁺¹⁺¹, x[3]=>x[2], x[2]=>x[3]) == ρ₁₊₁₊₁(τ₂)*¹⁺¹⁺¹


######
# orthogonal
######

n = 5
@polyvar x[1:n]


# 2 = 2
p² = 1
@test gelfand(p², 1) == p²

# 2 = 1+1
p¹⁺¹ = x[2]-x[1]
@test gelfand(p¹⁺¹, 1) == -p¹⁺¹


# 3 = 3
p³ = p²
@test gelfand(p³, 1) == p³
@test gelfand(p³, 2) == 2p³

# 3 = 2+1
p²⁺¹₁ = x[2]-x[1]
p²⁺¹₂ = x[3]-x[1] + x[3]-x[2]
@test gelfand(p²⁺¹₁, 1) == -p²⁺¹₁
@test gelfand(p²⁺¹₁, 2) == p²⁺¹₁
@test gelfand(p²⁺¹₂, 1) == p²⁺¹₂
@test gelfand(p²⁺¹₂, 2) == -p²⁺¹₂

p²⁺¹₁₂ = (x[2]-x[1])*x[3]          # [1 2; 3] , [1 3; 2]
p²⁺¹₂₂ = (x[3]-x[1])*x[2] + (x[3]-x[2])*x[1]
# (x[3]^2-x[1]^2 + x[3]^2-x[2]^2)    # [1 3; 2] , [1 3; 2]

@test gelfand(p²⁺¹₁₂, 1) == -p²⁺¹₁₂
@test gelfand(p²⁺¹₁₂, 2) == p²⁺¹₁₂
@test gelfand(p²⁺¹₂₂, 1) == p²⁺¹₂₂
@test gelfand(p²⁺¹₂₂, 2) == -p²⁺¹₂₂

# 3 = 1+1+1

p¹⁺¹⁺¹ = p¹⁺¹ * (x[3]-x[1])*(x[3]-x[2])
@test gelfand(p¹⁺¹⁺¹, 1) == -p¹⁺¹⁺¹
@test gelfand(p¹⁺¹⁺¹, 2) == -2p¹⁺¹⁺¹

####
# n = 4
####

# 4 = 4
p⁴ = p³
@test gelfand(p⁴, 1) == p⁴
@test gelfand(p⁴, 2) == 2p⁴
@test gelfand(p⁴, 3) == 3p⁴

# 2 = 3+1
p³⁺¹₁ = p²⁺¹₁  # x[2]-x[1] # [1 3 4; 2 0 0]
p³⁺¹₂ = p²⁺¹₂ # x[3]-x[1] + x[3]-x[2] # [1 2 4; 3]
p³⁺¹₃ = x[4]-x[1] + x[4]-x[2] + x[4]-x[3] # [1 2 3; 4]
@test gelfand(p³⁺¹₁, 1) == -p³⁺¹₁
@test gelfand(p³⁺¹₁, 2) == p³⁺¹₁
@test gelfand(p³⁺¹₁, 3) == 2p³⁺¹₁
@test gelfand(p³⁺¹₂, 1) == p³⁺¹₂
@test gelfand(p³⁺¹₂, 2) == -p³⁺¹₂
@test gelfand(p³⁺¹₂, 3) == 2p³⁺¹₂
@test gelfand(p³⁺¹₃, 1) == p³⁺¹₃
@test gelfand(p³⁺¹₃, 2) == 2p³⁺¹₃
@test gelfand(p³⁺¹₃, 3) == -p³⁺¹₃




# 4 = 2+2

p²⁺²₁ = p²⁺¹₁*x[4] - p²⁺¹₁₂  # (x[2]-x[1]) * (x[4]-x[3]) # [1 3; 2 4]
p²⁺²₂ = p²⁺¹₂*x[4] - p²⁺¹₂₂  # 2*(x[3]-x[1])*(x[4]-x[2]) - p²⁺²₁ #+p²⁺²₁
@test gelfand(p²⁺²₁, 1) == -p²⁺²₁
@test gelfand(p²⁺²₁, 2) == p²⁺²₁
@test gelfand(p²⁺²₁, 3) == 0
@test gelfand(p²⁺²₂, 1) == p²⁺²₂
@test gelfand(p²⁺²₂, 2) == -p²⁺²₂
@test gelfand(p²⁺²₂, 3) == 0



@test gelfand(p²⁺¹₁*x[4], 3) == p²⁺¹₁*x[4] + p²⁺¹₁₂
@test gelfand(p²⁺¹₁₂, 3) == p²⁺¹₁*x[4] + p²⁺¹₁₂


# 4 = 2+1+1

p²⁺¹⁺¹₁ = p²⁺¹₁*(3x[4]*(x[1]+x[2]+x[3]-x[4])-2*(x[1]x[2]+x[1]x[3]+x[2]x[3])) - p²⁺¹₁₂*(x[4]-x[1] + x[4]-x[2] +x[4]-x[3]) # [1 3; 2; 4]
p²⁺¹⁺¹₂ = p²⁺¹₂*(3x[4]*(x[1]+x[2]+x[3]-x[4])-2*(x[1]x[2]+x[1]x[3]+x[2]x[3])) - p²⁺¹₂₂*(x[4]-x[1] + x[4]-x[2] +x[4]-x[3]) # [1 2; 3; 4]
p²⁺¹⁺¹₃ = p¹⁺¹⁺¹  # [1 4 ; 2; 3]


# (x[4]-x[2])*(x[4]-x[1])*(x[2]-x[1])
# (x[4]-x[3])*(x[4]-x[1])*(x[3]-x[1])


x[4]-x[1] + x[4]-x[2] + x[4]-x[3] # [1 2; 3; 4]


@test gelfand(p²⁺¹⁺¹₁, 1) == -p²⁺¹⁺¹₁
@test gelfand(p²⁺¹⁺¹₁, 2) == p²⁺¹⁺¹₁
@test gelfand(p²⁺¹⁺¹₁, 3) == -2p²⁺¹⁺¹₁

@test gelfand(p²⁺¹⁺¹₂, 1) == p²⁺¹⁺¹₂
@test gelfand(p²⁺¹⁺¹₂, 2) == -p²⁺¹⁺¹₂
@test gelfand(p²⁺¹⁺¹₂, 3) == -2p²⁺¹⁺¹₂

@test gelfand(p²⁺¹⁺¹₃, 1) == -p²⁺¹⁺¹₃
@test gelfand(p²⁺¹⁺¹₃, 2) == -2p²⁺¹⁺¹₃
@test gelfand(p²⁺¹⁺¹₃, 3) == p²⁺¹⁺¹₃

p¹⁺¹⁺¹⁺¹ = p¹⁺¹⁺¹ * (x[4]-x[1])*(x[4]-x[2])*(x[4]-x[3])

@test gelfand(p¹⁺¹⁺¹⁺¹, 1) == -p¹⁺¹⁺¹⁺¹
@test gelfand(p¹⁺¹⁺¹⁺¹, 2) == -2p¹⁺¹⁺¹⁺¹
@test gelfand(p¹⁺¹⁺¹⁺¹, 3) == -3p¹⁺¹⁺¹⁺¹



# derivation
# @test gelfand(p³₁*x[4]^2, 3) == 

@polyvar a[1:5]

q = -3*(x[4]-x[2])*(x[4]-x[1])*(x[2]-x[1]) + (x[3]-x[2])*(x[3]-x[1])*(x[2]-x[1]) # + a[3]*(x[4]-x[3])*(x[4]-x[1])*(x[3]-x[1])


@test q == p³₁₂*(x[3]-x[2]-x[1]) + p³₁ * (3*x[4]*(x[2]+x[1]-x[4]) - 2x[1]x[2])


@test gelfand(q, 1) == -q
@test gelfand(q, 2) == q
@test gelfand(q, 3) == -2q

@test q == -3p³₁*x[4]^2 + 3p³₁*x[4]*(x[1]+x[2]+x[3]) - 3p³₁₂*x[4] + p³₁₂*(x[1]+x[2]+x[3]) - p³₁*(x[1]+x[2]+x[3])^2 + p³₁*(x[1]^2+x[2]^2+x[3]^2)
@test q == p³₁*(3x[4]*(x[1]+x[2]+x[3]-x[4])-2*(x[1]x[2]+x[1]x[3]+x[2]x[3])) - p³₁₂*(x[4]-x[1] + x[4]-x[2] +x[4]-x[3])

3gelfand(p³₁*(3x[4]*(x[1]+x[2]+x[3]-x[4])-2*(x[1]x[2]+x[1]x[3]+x[2]x[3])),3) - 
(-5p³₁*(3x[4]*(x[1]+x[2]+x[3]-x[4])-2*(x[1]x[2]+x[1]x[3]+x[2]x[3])))

11p³₁₂*(x[4]-x[1] + x[4]-x[2] +x[4]-x[3])



3x[4]*(x[1]+x[2]+x[3]-x[4])-2*(x[1]x[2]+x[1]x[3]+x[2]x[3]) + 
    ((x[4]-x[1])*(x[4]-x[2]) + (x[4]-x[1])*(x[4]-x[3]) + (x[4]-x[2])*(x[4]-x[3]))

(x[4]-x[2])*(x[3] - x[1])

(x[4]-x[1])*(x[4]-x[2]) + (x[4]-x[1])*(x[4]-x[3]) + (x[4]-x[2])*(x[4]-x[3]) + (x[1]-x[2])*(x[1]-x[3])  + (x[2]-x[1])*(x[2]-x[3]) +  (x[3]-x[1])*(x[3]-x[2])



+x[2]+x[3])
x[3]*x[1]




gelfand(p³₂*x[4]^2, 3)
gelfand(p³₂*x[4]*(x[1]+x[2]+x[3]), 3)
gelfand(p³₂*(x[1]+x[2]+x[3])^2, 3)
gelfand(p³₂*(x[1]^2+x[2]^2+x[3]^2), 3)
gelfand(p³₂₂*x[4], 3)
gelfand(p³₂₂*(x[1]+x[2]+x[3]), 3)




 gelfand(p³₁*x[4]^2, 3) # -p³₁*x[4]^2
p³₁₂ * x[4]
@test gelfand(p³₁₂, 3) == p³₁*x[4] + p³₁₂
@test gelfand(p³₁*(x[1]+x[2]+x[3]+x[4]), 3) == 2p³₁*(x[1]+x[2]+x[3]+x[4])

@polyvar c d

c = -1; d = -1/2
gelfand(p³₁₂+c*p³₁*x[4]+d*p³₁*(x[1]+x[2]+x[3]+x[4]), 3) + 2*(p³₁₂+c*p³₁*x[4]+d*p³₁*(x[1]+x[2]+x[3]+x[4]))

# 1-4d+3c == 0
# 3+4d+c == 0

@test gelfand(p³₁₂+p³₁*x[4], 3) == 2*(p³₁*x[4] + p³₁₂)


#####
# n = 5
#####

p⁵ = p⁴
@test gelfand(p⁵, 1) == p⁵
@test gelfand(p⁵, 2) == 2p⁵
@test gelfand(p⁵, 3) == 3p⁵
@test gelfand(p⁵, 4) == 4p⁵

# [1 3 4 5; 2]
p³⁺¹₁

# [1 2 3 4; 5]
p⁴⁺¹₁ = sum(x[5]-x[k] for k = 1:4)
@test gelfand(p⁴⁺¹₁, 1) == p⁴⁺¹₁
@test gelfand(p⁴⁺¹₁, 2) == 2p⁴⁺¹₁
@test gelfand(p⁴⁺¹₁, 3) == 3p⁴⁺¹₁
@test gelfand(p⁴⁺¹₁, 4) == -p⁴⁺¹₁
