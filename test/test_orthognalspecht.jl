using NumericalRepresentationTheory, DynamicPolynomials, Permutations
using DynamicPolynomials: Monomial

gelfand(p, x, m) = sum(subs(p, x[k] => x[m+1], x[m+1] => x[k]) for k=1:m)
laplacian(p, x, k=1) = sum(differentiate.(p, x, k))

function _polydiff(q, x, k)
    isempty(x) && return q
    _polydiff(differentiate(q, x[1], k[1]), x[2:end], k[2:end])
end
polydiff(q, m::Monomial) = _polydiff(q, m.vars, m.z)
polydiff(q, p) = sum(t.coefficient*polydiff(q, t.monomial) for t in terms(p))


######
# orthogonal
######

n = 5
@polyvar x[1:n]

p² = 1
p¹⁺¹ = Δ² = x[2]-x[1]

A₃ = x[1]+x[2]-2x[3]
B₃ = 2*(x[1]-x[3])*(x[2]-x[3]) - (x[1]-x[2])^2
C₃ = (x[3]-x[1])*(x[3]-x[2])
p³ = p²
p²⁺¹₁ = p¹⁺¹ #  [1 3; 2]
p²⁺¹₂ = A₃   # [1 2; 3]
p¹⁺¹⁺¹ = Δ³ = p¹⁺¹ * C₃
p²⁺¹₁₂ =  p¹⁺¹*A₃       # [1 3; 2] , [1 2; 3]
p²⁺¹₂₂ = B₃    # [1 2; 3] , [1 3; 2]


A₄ = -(x[4]-x[1] + x[4]-x[2] + x[4]-x[3])
B₄ = -9x[4]^2 + 6x[4]*(x[1]+x[2]+x[3]) - 2*(x[1]+x[2]+x[3])^2 + 3*(x[1]^2+x[2]^2+x[3]^2)
C₄ = -3x[4]^3 + 3x[3]x[4]^2 + x[3]^2*x[4] + x[3]^3 + 3x[2]x[4]^2 - 4x[2]x[3]x[4] - 2x[2]x[3]^2 + x[2]^2*x[4] - 2x[2]^2*x[3] + x[2]^3 + 3x[1]x[4]^2 - 4x[1]x[3]x[4] - 2x[1]x[3]^2 - 4x[1]x[2]x[4] + 12x[1]x[2]x[3] - 2x[1]x[2]^2 + x[1]^2*x[4] - 2x[1]^2*x[3] - 2x[1]^2*x[2] + x[1]^3
D₄ = 6x[4]^2 - 4x[3]x[4] - x[3]^2 - 4x[2]x[4] + 3x[2]x[3] - x[2]^2 - 4x[1]x[4] + 3x[1]x[3] + 3x[1]x[2] - x[1]^2
E₄ = -9x[4]^2 + 6x[3]x[4] - x[3]^2 + 6x[2]x[4] - 2x[2]x[3] - x[2]^2 + 6x[1]x[4] - 2x[1]x[3] - 2x[1]x[2] - x[1]^2
F₄ = -9x[4]^3 + 9x[3]x[4]^2 + 3x[3]^2*x[4] + x[3]^3 + 9x[2]x[4]^2 - 12x[2]x[3]x[4] - 3x[2]x[3]^2 + 3x[2]^2*x[4] - 3x[2]^2*x[3] + x[2]^3 + 9x[1]x[4]^2 - 12x[1]x[3]x[4] - 3x[1]x[3]^2 - 12x[1]x[2]x[4] + 24x[1]x[2]x[3] - 3x[1]x[2]^2 + 3x[1]^2*x[4] - 3x[1]^2*x[3] - 3x[1]^2*x[2] + x[1]^3
G₄ = x[1]^2+x[2]^2+x[3]^2 - 3x[4]^2 + 2(x[3]x[4] + x[2]x[4] - x[2]x[3] + x[1]x[4] - x[1]x[3] - x[1]x[2])
H₄ = prod(x[4] .- x[1:3])
I₄ = 6x[4]^3 - 6x[3]x[4]^2 - 6x[3]^2*x[4] - 2x[3]^3 - 6x[2]x[4]^2 + 12x[2]x[3]x[4] + 6x[2]x[3]^2 - 6x[2]^2*x[4] + 6x[2]^2*x[3] - 2x[2]^3 - 6x[1]x[4]^2 + 12x[1]x[3]x[4] + 6x[1]x[3]^2 + 12x[1]x[2]x[4]- 36x[1]x[2]x[3] + 6x[1]x[2]^2 - 6x[1]^2*x[4] + 6x[1]^2*x[3] + 6x[1]^2*x[2] - 2x[1]^3


p⁴ = p³
p³⁺¹₁ = p²⁺¹₁  # x[2]-x[1] # [1 3 4; 2 0 0]
p³⁺¹₂ = p²⁺¹₂ # x[3]-x[1] + x[3]-x[2] # [1 2 4; 3]
p³⁺¹₃ = A₄ # [1 2 3; 4]
p²⁺²₁ = p²⁺¹₁*A₄ - p²⁺¹₁₂  # (x[2]-x[1]) * (x[4]-x[3]) # [1 3; 2 4]
p²⁺²₂ = p²⁺¹₂*A₄ + p²⁺¹₂₂
p²⁺¹⁺¹₁ = p²⁺¹₁*B₄ - p²⁺¹₁₂*A₄ # [1 3; 2; 4]
p²⁺¹⁺¹₂ = p²⁺¹₂*B₄ + p²⁺¹₂₂*A₄ # [1 2; 3; 4]
p²⁺¹⁺¹₃ = p¹⁺¹⁺¹  # [1 4 ; 2; 3]
p¹⁺¹⁺¹⁺¹ = Δ⁴ = p¹⁺¹⁺¹ * H₄


p³⁺¹₁₂ = A₄*p²⁺¹₁ + 2p²⁺¹₁₂
p³⁺¹₂₂ = A₄*p²⁺¹₂ - 2p²⁺¹₂₂
p³⁺¹₃₂ = G₄
p²⁺²₁₂ = p²⁺¹₁*C₄ + p²⁺¹₁₂*D₄
p²⁺²₂₂ = p²⁺¹₂*C₄ - p²⁺¹₂₂*D₄
p²⁺¹⁺¹₁₂ = p²⁺¹₁*F₄ + E₄*p²⁺¹₁₂
p²⁺¹⁺¹₂₂ = p²⁺¹₂*F₄ - E₄*p²⁺¹₂₂
p²⁺¹⁺¹₃₂ = p¹⁺¹⁺¹*A₄

a = x[1]-x[2]
b = x[3]-x[4]
c = x[1]+x[2]-x[3]-x[4]
p³⁺¹₁₃ = p²⁺¹₁*B₄ + 5p²⁺¹₁₂*A₄
p³⁺¹₂₃ = p²⁺¹₂*B₄ - 5p²⁺¹₂₂*A₄
p³⁺¹₃₃ = I₄
p²⁺¹⁺¹₁₃ = a*(a^4+5b^4+c^4-6a^2*b^2+8a^2*b*c-2a^2*c^2+8b^3*c-6b^2*c^2-8b*c^3)
p²⁺¹⁺¹₂₃ = b*(5a^4+b^4+c^4-6a^2*b^2-6a^2*c^2-2b^2*c^2)
p²⁺¹⁺¹₃₃ = a*(a-b+c)*(a+b-c)*(a^2-5b^2-6b*c-c^2)


A₅ = sum( x[1:4] .- x[5])

p⁵ = p⁴
p⁴⁺¹₁ = p³⁺¹₁ # [1 3 4 5; 2]
p⁴⁺¹₂ = p³⁺¹₂  # [1 2 4 5; 3]
p⁴⁺¹₃ = p³⁺¹₃  # [1 2 3 5; 4]
p⁴⁺¹₄ = A₅
p³⁺²₁ = p²⁺²₁ # [1 3 5; 2 4]
p³⁺²₂ = p²⁺²₂ # [1 2 5; 3 4]
p³⁺²₃ = -3p³⁺¹₁*A₅ + p³⁺¹₁₂  # [1 3 4; 2 5]
p³⁺²₄ = -3p³⁺¹₂*A₅ + p³⁺¹₂₂ # [1 2 4; 3 5]
p³⁺²₅ = p³⁺¹₃*A₅ - p³⁺¹₃₂ # [1 2 3; 4 5]



gelfand(p³⁺¹₃*A₅, x, 4)
gelfand(p³⁺¹₃₂, x, 4)

err = gelfand(p³⁺¹₁*(-3sum(x[1:4]) + (-2*(-3)+6)*x[5]) + p³⁺¹₁₂, x, 4)
t = x[2]^2
coefficient(err, t)
[coefficient(err, t*α[k]) for k=1:2]
[coefficient(err, t*β[k]) for k=1:1]


e_1 = sum(x[1:3] .^3)
e_2 = sum(x[1:3] .^2)*sum(x[1:3])
e_3 = sum(x[1:3])^3
e_4 = sum(x[1:3] .^ 2) * x[4]
e_5 = sum(x[1:3])^2 * x[4]
e_6 = sum(x[1:3]) * x[4]^2
e_7 = x[4]^3

f_1 = sum(x[1:3] .^2)
f_2 = sum(x[1:3])^2
f_3 = sum(x[1:3])*x[4]
f_4 = x[4]^2

# derive p²⁺¹⁺¹₁₃



# derive p³⁺¹₂₃

err = 3p³⁺¹₂₃ - 2*(p²⁺¹₂*B₄ - 5A₄*p²⁺¹₂₂)
t = x[2]*x[5]
coefficient(err, t)
[coefficient(err, t*α[k]) for k=1:4]
[coefficient(err, t*β[k]) for k=1:2]

#    β[1] == -32//9

# derive p²⁺¹⁺¹₂₂

p²⁺¹⁺¹₂₂
β_2 = -4//18
err = 9p²⁺¹⁺¹₂₂ - 2*(p²⁺¹₂*(-F₄) + E₄*p²⁺¹₂₂)

E₄

err = 9p²⁺¹⁺¹₂₂ - 2*(p²⁺¹₂*dot([-12,15,-4,-9,6,-9,9],[e_1,e_2,e_3,e_4,e_5,e_6,e_7]) + dot([0,-1,6,-9],[f_1,f_2,f_3,f_4])*p²⁺¹₂₂)

t = x[1]x[2]*x[3]^2
coefficient(err, t)
[coefficient(err, t*α[k]) for k=1:7]
[coefficient(err, t*β[k]) for k=1:4]

α[2] == 2//3 - 3α[3]

 β[3] == 4//3

err = p²⁺¹⁺¹₂₂ - (p²⁺¹₂*dot([α[1],α[2],α[3],α[4],α[5],α[6],α[7]],[e_1,e_2,e_3,e_4,e_5,e_6,e_7]) + dot([β[1],β[2],β[3],β[4]],[f_1,f_2,f_3,f_4])*p²⁺¹₂₂)




# derive p²⁺¹⁺¹₁₂

err = 3p²⁺¹⁺¹₁₂ - 2*(p²⁺¹₁*F₄ + E₄*p²⁺¹₁₂)
B₄

E₄

err = 3p²⁺¹⁺¹₁₂ - 2*(p²⁺¹₁*dot([12,-15,4,9,-6,9,-9],[e_1,e_2,e_3,e_4,e_5,e_6,e_7]) + dot([0,-1,6,-9],[f_1,f_2,f_3,f_4])*p²⁺¹₁₂)
t = x[2]^3*x[3]
coefficient(err, t)
[coefficient(err, t*α[k]) for k=1:7]
[coefficient(err, t*β[k]) for k=1:4]

-12+2α[4] == 0

# derive p²⁺²₁₂


@polyvar α[1:7]
@polyvar β[1:4]


@test 3p²⁺²₁₂ == 2*(p²⁺¹₁*C₄ + p²⁺¹₁₂*D₄)

err = 3p²⁺²₁₂ - (p²⁺¹₁*dot([14,-16, 4,6,-4 ,6,-6],[e_1,e_2,e_3,e_4,e_5,e_6,e_7]) + dot([-5,3,-8,12],[f_1,f_2,f_3,f_4])*p²⁺¹₁₂)
t = x[1]x[2]^2*x[3]
coefficient(err, t)
[coefficient(err, t*α[k]) for k=1:7]
[coefficient(err, t*β[k]) for k=1:4]
8//3 + β[1] == β[2]


@polyvar α[1:6]
a*(a^2+3c^2 -9b^2)

# 2 = 2
@test gelfand(p², x, 1) == p²
@test all(iszero, [laplacian(p², x[1:2], k) for k=1:2])

# 2 = 1+1

@test gelfand(p¹⁺¹, x, 1) == -p¹⁺¹
@test all(iszero, [laplacian(p¹⁺¹, x[1:2], k) for k=1:2])

# 3 = 3
@test gelfand(p³, x, 1) == p³
@test gelfand(p³, x, 2) == 2p³
@test all(iszero, [laplacian(p³, x[1:3], k) for k=1:3])

# 3 = 2+1
@test gelfand(p²⁺¹₁, x, 1) == -p²⁺¹₁
@test gelfand(p²⁺¹₁, x, 2) == p²⁺¹₁
@test gelfand(p²⁺¹₂, x, 1) == p²⁺¹₂
@test gelfand(p²⁺¹₂, x, 2) == -p²⁺¹₂




@test gelfand(p²⁺¹₁₂, x, 1) == -p²⁺¹₁₂
@test gelfand(p²⁺¹₁₂, x, 2) == p²⁺¹₁₂
@test gelfand(p²⁺¹₂₂, x, 1) == p²⁺¹₂₂
@test gelfand(p²⁺¹₂₂, x, 2) == -p²⁺¹₂₂
for p in (p²⁺¹₁, p²⁺¹₂, p²⁺¹₁₂, p²⁺¹₂₂)
    @test all(iszero, [laplacian(p, x[1:3], k) for k=1:3])
end

# 3 = 1+1+1

@test gelfand(p¹⁺¹⁺¹, x, 1) == -p¹⁺¹⁺¹
@test gelfand(p¹⁺¹⁺¹, x, 2) == -2p¹⁺¹⁺¹
for p in (p¹⁺¹⁺¹,)
    @test all(iszero, [laplacian(p, x[1:3], k) for k=1:3])
end

# duality


# p²⁺¹₁(D)*Δ³
@test polydiff(Δ³, p²⁺¹₁) == differentiate(Δ³, x[2]) - differentiate(Δ³, x[1]) == p²⁺¹₂₂
# p²⁺¹₂(D)*Δ
@test polydiff(Δ³, p²⁺¹₂) == -(2differentiate(Δ³, x[3]) - differentiate(Δ³, x[1]) - differentiate(Δ³, x[2])) == 3p²⁺¹₁₂


# quadratic form for higher-order


@test p²⁺¹₁₂ == x[1:3]'*[-1 0 1; 0 1 -1; 1 -1 0]*x[1:3]
@test p²⁺¹₂₂ == x[1:3]'*[-1 2 -1; 2 -1 -1; -1 -1 2]*x[1:3]

####
# n = 4
####

# 4 = 4

@test gelfand(p⁴, x, 1) == p⁴
@test gelfand(p⁴, x, 2) == 2p⁴
@test gelfand(p⁴, x, 3) == 3p⁴

for p in (p⁴,)
    @test all(iszero, [laplacian(p, x[1:4], k) for k=1:4])
end



# 2 = 3+1
@test gelfand(p³⁺¹₁, x, 1) == -p³⁺¹₁
@test gelfand(p³⁺¹₁, x, 2) == p³⁺¹₁
@test gelfand(p³⁺¹₁, x, 3) == 2p³⁺¹₁
@test gelfand(p³⁺¹₂, x, 1) == p³⁺¹₂
@test gelfand(p³⁺¹₂, x, 2) == -p³⁺¹₂
@test gelfand(p³⁺¹₂, x, 3) == 2p³⁺¹₂
@test gelfand(p³⁺¹₃, x, 1) == p³⁺¹₃
@test gelfand(p³⁺¹₃, x, 2) == 2p³⁺¹₃
@test gelfand(p³⁺¹₃, x, 3) == -p³⁺¹₃

@test gelfand(p³⁺¹₁₂, x, 1) == -p³⁺¹₁₂
@test gelfand(p³⁺¹₁₂, x, 2) == p³⁺¹₁₂
@test gelfand(p³⁺¹₁₂, x, 3) == 2p³⁺¹₁₂
@test gelfand(p³⁺¹₂₂, x, 1) == p³⁺¹₂₂
@test gelfand(p³⁺¹₂₂, x, 2) == -p³⁺¹₂₂
@test gelfand(p³⁺¹₂₂, x, 3) == 2p³⁺¹₂₂
@test gelfand(p³⁺¹₃₂, x, 1) == p³⁺¹₃₂
@test gelfand(p³⁺¹₃₂, x, 2) == 2p³⁺¹₃₂
@test gelfand(p³⁺¹₃₂, x, 3) == -p³⁺¹₃₂


@test gelfand(p³⁺¹₁₃, x, 1) == -p³⁺¹₁₃
@test gelfand(p³⁺¹₁₃, x, 2) == p³⁺¹₁₃
@test gelfand(p³⁺¹₁₃, x, 3) == 2p³⁺¹₁₃
@test gelfand(p³⁺¹₂₃, x, 1) == p³⁺¹₂₃
@test gelfand(p³⁺¹₂₃, x, 2) == -p³⁺¹₂₃
@test gelfand(p³⁺¹₂₃, x, 3) == 2p³⁺¹₂₃
@test gelfand(p³⁺¹₃₃, x, 1) == p³⁺¹₃₃
@test gelfand(p³⁺¹₃₃, x, 2) == 2p³⁺¹₃₃
@test gelfand(p³⁺¹₃₃, x, 3) == -p³⁺¹₃₃



for p in (p³⁺¹₁,p³⁺¹₂,p³⁺¹₃,p³⁺¹₁₂,p³⁺¹₂₂,p³⁺¹₃₂,p³⁺¹₁₃,p³⁺¹₂₃,p³⁺¹₃₃)
    @test all(iszero, [laplacian(p, x[1:4], k) for k=1:4])
end


@polyvar a b c

@test p³⁺¹₂₂ == -3*(2x[3]x[4]-2x[3]^2-x[2]x[4]+x[2]x[3]+x[2]^2-x[1]x[4]+x[1]x[3]-2x[1]x[2]+x[1]^2)


a^2-b^2+b*c

@test p³⁺¹₁₂ == 3*(x[1]+x[2]-x[3]-x[4])*p²⁺¹₁

# (x[1]+x[2]-x[3]-x[4])*p²⁺¹₁       # [1 3 4; 2] , [1 2; 3]
p²⁺¹₂₂ = 2*(x[1]-x[3])*(x[2]-x[3]) - (x[1]-x[2])^2    # [1 3; 2] , [1 3; 2]

x₂x₄b + 2x₂x₃c + x₂x₃a - x₂²c + x₂²a - x₁x₄b - 2x₁x₃c - x₁x₃a + x₁²c - x₁²a
c = -2/3
b = -1
(x[1]+x[2]+x[3]-3x[4])*p²⁺¹₁ - 2*p²⁺¹₁₂ - 3p³⁺¹₁₂




a^2-b^2-2b*c

p²⁺¹₁
@polyvar α β
α = -4/3
β = α+1
α*(x[1]+x[2]+x[3]-3x[4])*p²⁺¹₁ - β*p²⁺¹₁₂












# 4 = 2+2

# p²⁺²₂ =    p²⁺¹₂*x[4] - p²⁺¹₂₂  # 2*(x[3]-x[1])*(x[4]-x[2]) - p²⁺²₁ #+p²⁺²₁


@test p²⁺²₁ == -3a*b
@test 2p²⁺²₂ == -3*(a^2 + b^2 - c^2)



@test gelfand(p²⁺²₁, x, 1) == -p²⁺²₁
@test gelfand(p²⁺²₁, x, 2) == p²⁺²₁
@test gelfand(p²⁺²₁, x, 3) == 0
@test gelfand(p²⁺²₂, x, 1) == p²⁺²₂
@test gelfand(p²⁺²₂, x, 2) == -p²⁺²₂
@test gelfand(p²⁺²₂, x, 3) == 0

@test gelfand(p²⁺²₁₂, x, 1) == -p²⁺²₁₂
@test gelfand(p²⁺²₁₂, x, 2) == p²⁺²₁₂
@test gelfand(p²⁺²₁₂, x, 3) == 0
@test gelfand(p²⁺²₂₂, x, 1) == p²⁺²₂₂
@test gelfand(p²⁺²₂₂, x, 2) == -p²⁺²₂₂
@test gelfand(p²⁺²₂₂, x, 3) == 0



for p in (p²⁺²₁,p²⁺²₂,p²⁺²₁₂,p²⁺²₂₂)
    @test all(iszero, [laplacian(p, x[1:4], k) for k=1:4])
end

# 4 = 2+1+1







# (x[4]-x[2])*(x[4]-x[1])*(x[2]-x[1])
# (x[4]-x[3])*(x[4]-x[1])*(x[3]-x[1])


x[4]-x[1] + x[4]-x[2] + x[4]-x[3] # [1 2; 3; 4]


@test gelfand(p²⁺¹⁺¹₁, x, 1) == -p²⁺¹⁺¹₁
@test gelfand(p²⁺¹⁺¹₁, x, 2) == p²⁺¹⁺¹₁
@test gelfand(p²⁺¹⁺¹₁, x, 3) == -2p²⁺¹⁺¹₁

@test gelfand(p²⁺¹⁺¹₂, x, 1) == p²⁺¹⁺¹₂
@test gelfand(p²⁺¹⁺¹₂, x, 2) == -p²⁺¹⁺¹₂
@test gelfand(p²⁺¹⁺¹₂, x, 3) == -2p²⁺¹⁺¹₂

@test gelfand(p²⁺¹⁺¹₃, x, 1) == -p²⁺¹⁺¹₃
@test gelfand(p²⁺¹⁺¹₃, x, 2) == -2p²⁺¹⁺¹₃
@test gelfand(p²⁺¹⁺¹₃, x, 3) == p²⁺¹⁺¹₃

@test gelfand(p²⁺¹⁺¹₁₂, x, 1) == -p²⁺¹⁺¹₁₂
@test gelfand(p²⁺¹⁺¹₁₂, x, 2) == p²⁺¹⁺¹₁₂
@test gelfand(p²⁺¹⁺¹₁₂, x, 3) == -2p²⁺¹⁺¹₁₂

@test gelfand(p²⁺¹⁺¹₂₂, x, 1) == p²⁺¹⁺¹₂₂
@test gelfand(p²⁺¹⁺¹₂₂, x, 2) == -p²⁺¹⁺¹₂₂
@test gelfand(p²⁺¹⁺¹₂₂, x, 3) == -2p²⁺¹⁺¹₂₂

@test gelfand(p²⁺¹⁺¹₃₂, x, 1) == -p²⁺¹⁺¹₃₂
@test gelfand(p²⁺¹⁺¹₃₂, x, 2) == -2p²⁺¹⁺¹₃₂
@test gelfand(p²⁺¹⁺¹₃₂, x, 3) == p²⁺¹⁺¹₃₂


@test gelfand(p²⁺¹⁺¹₁₃, x, 1) == -p²⁺¹⁺¹₁₃
@test gelfand(p²⁺¹⁺¹₁₃, x, 2) == p²⁺¹⁺¹₁₃
@test gelfand(p²⁺¹⁺¹₁₃, x, 3) == -2p²⁺¹⁺¹₁₃

@test gelfand(p²⁺¹⁺¹₂₃, x, 1) == p²⁺¹⁺¹₂₃
@test gelfand(p²⁺¹⁺¹₂₃, x, 2) == -p²⁺¹⁺¹₂₃
@test gelfand(p²⁺¹⁺¹₂₃, x, 3) == -2p²⁺¹⁺¹₂₃

@test gelfand(p²⁺¹⁺¹₃₃, x, 1) == -p²⁺¹⁺¹₃₃
@test gelfand(p²⁺¹⁺¹₃₃, x, 2) == -2p²⁺¹⁺¹₃₃
@test gelfand(p²⁺¹⁺¹₃₃, x, 3) == p²⁺¹⁺¹₃₃



for p in (p²⁺¹⁺¹₁,p²⁺¹⁺¹₂,p²⁺¹⁺¹₃,p²⁺¹⁺¹₁₂,p²⁺¹⁺¹₂₂,p²⁺¹⁺¹₃₂,p²⁺¹⁺¹₁₃,p²⁺¹⁺¹₂₃,p²⁺¹⁺¹₃₃)
    @test all(iszero, [laplacian(p, x[1:4], k) for k=1:4])
end


p¹⁺¹⁺¹⁺¹ = Δ⁴ = p¹⁺¹⁺¹ * (x[4]-x[1])*(x[4]-x[2])*(x[4]-x[3])

@test gelfand(p¹⁺¹⁺¹⁺¹, x, 1) == -p¹⁺¹⁺¹⁺¹
@test gelfand(p¹⁺¹⁺¹⁺¹, x, 2) == -2p¹⁺¹⁺¹⁺¹
@test gelfand(p¹⁺¹⁺¹⁺¹, x, 3) == -3p¹⁺¹⁺¹⁺¹



# duality

a = -9*x[4]^2 + 6*x[4]*(x[1]+x[2]+x[3]) - 2*(x[1]+x[2]+x[3])^2 + 3*(x[1]^2+x[2]^2+x[3]^2)
b = (x[1]+x[2]+x[3]-3x[4])

@test a == (x[1]-x[2])^2 + (x[2]-x[3])^2 + (x[3]-x[1])^2 - (x[4]-x[1] + x[4]-x[2] + x[4]-x[3])^2


# L_1 (p²⁺¹₁*a +  p²⁺¹₁₂*b) = p²⁺¹₁_1 a + p²⁺¹₁_2 a + p²⁺¹₁_3 a + p²⁺¹₁(a_1+a_2+a_3)

@test p²⁺¹⁺¹₁ ==   p²⁺¹₁*a +  p²⁺¹₁₂*b
@test p²⁺¹⁺¹₂ ==   p²⁺¹₂*a -  p²⁺¹₂₂*b

@test laplacian(p²⁺¹₁*a +  p²⁺¹₁₂*b,x) == p²⁺¹₁*laplacian(a,x) + p²⁺¹₁₂*laplacian(b,x)
@test laplacian(p²⁺¹₂*a -  p²⁺¹₂₂*b,x) == p²⁺¹₂*laplacian(a,x) + p²⁺¹₂₂*laplacian(b,x)
@test laplacian(a, x) == laplacian(b, x) == 0

@test laplacian(p²⁺¹₁*a, x, 2) == p²⁺¹₁*laplacian(a, x, 2) +
    2differentiate(p²⁺¹₁, x[1])*differentiate(a,x[1]) +
    2differentiate(p²⁺¹₁, x[2])*differentiate(a,x[2]) +
    2differentiate(p²⁺¹₁, x[3])*differentiate(a,x[3])

@test laplacian(p²⁺¹₁₂*b, x, 2) == p²⁺¹₁*laplacian(b, x, 2) +
    2differentiate(p²⁺¹₁, x[1])*differentiate(b,x[1]) +
    2differentiate(p²⁺¹₁, x[2])*differentiate(b,x[2]) +
    2differentiate(p²⁺¹₁, x[3])*differentiate(b,x[3])

@test laplacian(p²⁺¹₂*a, x, 2) == p²⁺¹₂*laplacian(a, x, 2) +
    2differentiate(p²⁺¹₂, x[1])*differentiate(a,x[1]) +
    2differentiate(p²⁺¹₂, x[2])*differentiate(a,x[2]) +
    2differentiate(p²⁺¹₂, x[3])*differentiate(a,x[3])

@test laplacian(p²⁺¹₂₂*b, x, 2) == p²⁺¹₂*laplacian(b, x, 2) +
    2differentiate(p²⁺¹₂, x[1])*differentiate(b,x[1]) +
    2differentiate(p²⁺¹₂, x[2])*differentiate(b,x[2]) +
    2differentiate(p²⁺¹₂, x[3])*differentiate(b,x[3])


@test laplacian(b, x, 2) == 0



# derivation
# @test gelfand(p³₁*x[4]^2, 3) == 


q = -3*(x[4]-x[2])*(x[4]-x[1])*(x[2]-x[1]) + (x[3]-x[2])*(x[3]-x[1])*(x[2]-x[1]) # + a[3]*(x[4]-x[3])*(x[4]-x[1])*(x[3]-x[1])



@polyvar a b c d e
a = -2
b = 2
e = -1 # b - 3
d = 2 # 3 + 3e + b
c = -2 # -6-2a
3*2*q - (-2*p²⁺¹₁*(9*x[4]^2 + 3*a*x[4]*(x[1]+x[2]+x[3]) + b*(x[1]+x[2]+x[3])^2 + 3*e*(x[1]^2+x[2]^2+x[3]^2)) +  p²⁺¹₁₂*(3*c*x[4] + d*(x[1]+x[2]+x[3])))

@test 3*q == -p²⁺¹₁*(9*x[4]^2 + 3*a*x[4]*(x[1]+x[2]+x[3]) + b*(x[1]+x[2]+x[3])^2 + 3*e*(x[1]^2+x[2]^2+x[3]^2)) +  p²⁺¹₁₂*(x[1]+x[2]+x[3]-3x[4])


@test (9*x[4]^2 + 3*a*x[4]*(x[1]+x[2]+x[3]) + b*(x[1]+x[2]+x[3])^2 + 3*e*(x[1]^2+x[2]^2+x[3]^2)) ==
    9x[4]^2 - 6*x[4]*(x[1]+x[2]+x[3])


@test (9*x[4]^2 + 3*a*x[4]*(x[1]+x[2]+x[3]) + b*(x[1]+x[2]+x[3])^2 + 3*e*(x[1]^2+x[2]^2+x[3]^2)) ==
    (9x[4]^2 - 6*x[4]*(x[1]+x[2]+x[3]) + 4*(x[1]x[2]+x[1]x[3]+x[2]x[3]) - (x[1]^2+x[2]^2+x[3]^2))

g = (z,w) -> 3x[4]*(x[4]-2*z) + 4z*w  - z^2

@test 2*(9x[4]^2 - 6*x[4]*(x[1]+x[2]+x[3]) + 4*(x[1]x[2]+x[1]x[3]+x[2]x[3]) - (x[1]^2+x[2]^2+x[3]^2)) == (
    g(x[1],x[2])+g(x[2],x[1]) + g(x[1],x[3])+g(x[3],x[1]) + g(x[2], x[3])+g(x[3],x[2])
    )

# (1), (2), (3)
# (12), (13), (23)

subs(p²⁺¹₁₂, x[1]=>x[4])
subs(p²⁺¹₁₂, x[2]=>x[4])

# x[4]-x[1] + x[4]-x[2] + x[4]-x[3]

(x[4]-x[2])*(x[4]-x[1])*(x[2]-x[1])
(x[4]-x[2])*(x[4]-x[1])
(x[3]-x[2])*(x[3]-x[1])

Δ⁴

-1 + 3b = e

2+6e-2d+6b == 0 # => d = 1+3e+3b
6+c+6a == 0 # c - -6-6a

p²⁺¹₁*(x[1]^2+x[2]^2+x[3]^2)

p²⁺¹₁₂*x[4]
p²⁺¹₁₂*(x[1]+x[2]+x[3])
p²⁺¹₁*x[4]^2
p²⁺¹₁*x[4]*(x[1]+x[2]+x[3])
p²⁺¹₁*(x[1]+x[2]+x[3])^2
p²⁺¹₁*(x[1]^2+x[2]^2+x[3]^2)


@test q == p²⁺¹₁₂*(x[3]-x[2]-x[1]) + p²⁺¹₁₂ * (3*x[4]*(x[2]+x[1]-x[4]) - 2x[1]x[2])


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


# n = 5
####

# 5 = 5

@test gelfand(p⁵, x, 1) == p⁵
@test gelfand(p⁵, x, 2) == 2p⁵
@test gelfand(p⁵, x, 3) == 3p⁵
@test gelfand(p⁵, x, 4) == 4p⁵

for p in (p⁴,)
    @test all(iszero, [laplacian(p, x[1:5], k) for k=1:5])
end



# 2 = 34+1
@test gelfand(p⁴⁺¹₁, x, 1) == -p⁴⁺¹₁
@test gelfand(p⁴⁺¹₁, x, 2) == p⁴⁺¹₁
@test gelfand(p⁴⁺¹₁, x, 3) == 2p⁴⁺¹₁
@test gelfand(p⁴⁺¹₁, x, 4) == 3p⁴⁺¹₁
@test gelfand(p⁴⁺¹₂, x, 1) == p⁴⁺¹₂
@test gelfand(p⁴⁺¹₂, x, 2) == -p⁴⁺¹₂
@test gelfand(p⁴⁺¹₂, x, 3) == 2p⁴⁺¹₂
@test gelfand(p⁴⁺¹₂, x, 4) == 3p⁴⁺¹₂
@test gelfand(p⁴⁺¹₃, x, 1) == p⁴⁺¹₃
@test gelfand(p⁴⁺¹₃, x, 2) == 2p⁴⁺¹₃
@test gelfand(p⁴⁺¹₃, x, 3) == -p⁴⁺¹₃
@test gelfand(p⁴⁺¹₃, x, 4) == 3p⁴⁺¹₃
@test gelfand(p⁴⁺¹₄, x, 1) == p⁴⁺¹₄
@test gelfand(p⁴⁺¹₄, x, 2) == 2p⁴⁺¹₄
@test gelfand(p⁴⁺¹₄, x, 3) == 3p⁴⁺¹₄
@test gelfand(p⁴⁺¹₄, x, 4) == -p⁴⁺¹₄

@test gelfand(p³⁺²₁, x, 1) == -p³⁺²₁
@test gelfand(p³⁺²₁, x, 2) == p³⁺²₁
@test gelfand(p³⁺²₁, x, 3) == 0
@test gelfand(p³⁺²₁, x, 4) == 2p³⁺²₁
@test gelfand(p²⁺²₂, x, 1) == p²⁺²₂
@test gelfand(p²⁺²₂, x, 2) == -p²⁺²₂
@test gelfand(p²⁺²₂, x, 3) == 0
@test gelfand(p²⁺²₂, x, 4) == 2p²⁺²₂
@test gelfand(p³⁺²₃, x, 1) == -p³⁺²₃
@test gelfand(p³⁺²₃, x, 2) == p³⁺²₃
@test gelfand(p³⁺²₃, x, 3) == 2p³⁺²₃
@test gelfand(p³⁺²₃, x, 4) == 0
@test gelfand(p³⁺²₄, x, 1) == p³⁺²₄
@test gelfand(p³⁺²₄, x, 2) == -p³⁺²₄
@test gelfand(p³⁺²₄, x, 3) == 2p³⁺²₄
@test gelfand(p³⁺²₄, x, 4) == 0
@test gelfand(p³⁺²₅, x, 1) == p³⁺²₅
@test gelfand(p³⁺²₅, x, 2) == 2p³⁺²₅
@test gelfand(p³⁺²₅, x, 3) == -p³⁺²₅
@test gelfand(p³⁺²₅, x, 4) == 0



# @test gelfand(p³⁺¹₁₂, x, 1) == -p³⁺¹₁₂
# @test gelfand(p³⁺¹₁₂, x, 2) == p³⁺¹₁₂
# @test gelfand(p³⁺¹₁₂, x, 3) == 2p³⁺¹₁₂
# @test gelfand(p³⁺¹₂₂, x, 1) == p³⁺¹₂₂
# @test gelfand(p³⁺¹₂₂, x, 2) == -p³⁺¹₂₂
# @test gelfand(p³⁺¹₂₂, x, 3) == 2p³⁺¹₂₂
# @test gelfand(p³⁺¹₃₂, x, 1) == p³⁺¹₃₂
# @test gelfand(p³⁺¹₃₂, x, 2) == 2p³⁺¹₃₂
# @test gelfand(p³⁺¹₃₂, x, 3) == -p³⁺¹₃₂


# @test gelfand(p³⁺¹₁₃, x, 1) == -p³⁺¹₁₃
# @test gelfand(p³⁺¹₁₃, x, 2) == p³⁺¹₁₃
# @test gelfand(p³⁺¹₁₃, x, 3) == 2p³⁺¹₁₃
# @test gelfand(p³⁺¹₂₃, x, 1) == p³⁺¹₂₃
# @test gelfand(p³⁺¹₂₃, x, 2) == -p³⁺¹₂₃
# @test gelfand(p³⁺¹₂₃, x, 3) == 2p³⁺¹₂₃
# @test gelfand(p³⁺¹₃₃, x, 1) == p³⁺¹₃₃
# @test gelfand(p³⁺¹₃₃, x, 2) == 2p³⁺¹₃₃
# @test gelfand(p³⁺¹₃₃, x, 3) == -p³⁺¹₃₃



# for p in (p³⁺¹₁,p³⁺¹₂,p³⁺¹₃,p³⁺¹₁₂,p³⁺¹₂₂,p³⁺¹₃₂,p³⁺¹₁₃,p³⁺¹₂₃,p³⁺¹₃₃)
#     @test all(iszero, [laplacian(p, x[1:4], k) for k=1:4])
# end
