using NumericalRepresentationTheory, DynamicPolynomials, Permutations
using DynamicPolynomials: Monomial



# compute p((1,m+1) . 𝐱) + p((2,m+1) . 𝐱) + … + p((m,m+1) . 𝐱)
gelfand(p, x, m) = sum(subs(p, x[k] => x[m+1], x[m+1] => x[k]) for k=1:m)
laplacian(p, x, k=1) = sum(differentiate.(p, x, k))

function _polydiff(q, x, k)
    isempty(x) && return q
    _polydiff(differentiate(q, x[1], k[1]), x[2:end], k[2:end])
end
polydiff(q, m::Monomial) = _polydiff(q, m.vars, m.z)
polydiff(q, p) = sum(t.coefficient*polydiff(q, t.monomial) for t in terms(p))
polydiff(q, k::Number) = k*q

# tests if content vectors of p match 𝐜
contentvectorequals(p, x, 𝐜) = all(gelfand.(Ref(p), Ref(x), 1:length(𝐜)) .== 𝐜 .* Ref(p))
contentvectorisapprox(p, x, 𝐜) = all(gelfand.(Ref(p), Ref(x), 1:length(𝐜)) .≈ 𝐜 .* Ref(p))

######
# orthogonal Specht polynomials, comment gives corresponding SYT
######
@polyvar x[1:6]

p² = 1 # [1 2]
p¹⁺¹ = Δ² = x[2]-x[1] # [1; 2]

A₃ = x[1]+x[2]-2x[3]
B₃ = 2*(x[1]-x[3])*(x[2]-x[3]) - (x[1]-x[2])^2
C₃ = (x[3]-x[1])*(x[3]-x[2])
p³ = p² # [1 2 3]
p²⁺¹₁ = p¹⁺¹ # [1 3; 2]     [1 3; 2]
p²⁺¹₂ = A₃   # [1 2; 3]     [1 3; 2]
p¹⁺¹⁺¹ = Δ³ = p¹⁺¹ * C₃ # [1; 2; 3]
p²⁺¹₁₂ =  p¹⁺¹*A₃       # [1 3; 2]      [1 2; 3]
p²⁺¹₂₂ = B₃    # [1 2; 3]       [1 2; 3]


A₄ = -(x[4]-x[1] + x[4]-x[2] + x[4]-x[3])
B₄ =  (x[1]-x[2])^2 + (x[1]-x[3])^2 + (x[2]-x[3])^2 - A₄^2 #-9x[4]^2 + 6x[4]*(x[1]+x[2]+x[3]) - 2*(x[1]+x[2]+x[3])^2 + 3*(x[1]^2+x[2]^2+x[3]^2)
C₄ = -3x[4]^3 + 3x[3]x[4]^2 + x[3]^2*x[4] + x[3]^3 + 3x[2]x[4]^2 - 4x[2]x[3]x[4] - 2x[2]x[3]^2 + x[2]^2*x[4] - 2x[2]^2*x[3] + x[2]^3 + 3x[1]x[4]^2 - 4x[1]x[3]x[4] - 2x[1]x[3]^2 - 4x[1]x[2]x[4] + 12x[1]x[2]x[3] - 2x[1]x[2]^2 + x[1]^2*x[4] - 2x[1]^2*x[3] - 2x[1]^2*x[2] + x[1]^3
D₄ = 6x[4]^2 - 4x[3]x[4] - x[3]^2 - 4x[2]x[4] + 3x[2]x[3] - x[2]^2 - 4x[1]x[4] + 3x[1]x[3] + 3x[1]x[2] - x[1]^2
E₄ = -9x[4]^2 + 6x[3]x[4] - x[3]^2 + 6x[2]x[4] - 2x[2]x[3] - x[2]^2 + 6x[1]x[4] - 2x[1]x[3] - 2x[1]x[2] - x[1]^2
F₄ = -9x[4]^3 + 9x[3]x[4]^2 + 3x[3]^2*x[4] + x[3]^3 + 9x[2]x[4]^2 - 12x[2]x[3]x[4] - 3x[2]x[3]^2 + 3x[2]^2*x[4] - 3x[2]^2*x[3] + x[2]^3 + 9x[1]x[4]^2 - 12x[1]x[3]x[4] - 3x[1]x[3]^2 - 12x[1]x[2]x[4] + 24x[1]x[2]x[3] - 3x[1]x[2]^2 + 3x[1]^2*x[4] - 3x[1]^2*x[3] - 3x[1]^2*x[2] + x[1]^3
G₄ = x[1]^2+x[2]^2+x[3]^2 - 3x[4]^2 + 2(x[3]x[4] + x[2]x[4] - x[2]x[3] + x[1]x[4] - x[1]x[3] - x[1]x[2])
H₄ = prod(x[4] .- x[1:3])
I₄ = 6x[4]^3 - 6x[3]x[4]^2 - 6x[3]^2*x[4] - 2x[3]^3 - 6x[2]x[4]^2 + 12x[2]x[3]x[4] + 6x[2]x[3]^2 - 6x[2]^2*x[4] + 6x[2]^2*x[3] - 2x[2]^3 - 6x[1]x[4]^2 + 12x[1]x[3]x[4] + 6x[1]x[3]^2 + 12x[1]x[2]x[4]- 36x[1]x[2]x[3] + 6x[1]x[2]^2 - 6x[1]^2*x[4] + 6x[1]^2*x[3] + 6x[1]^2*x[2] - 2x[1]^3
J₄ = -9x[4]^3 + 9x[3]x[4]^2 + 2x[3]^2*x[4] + 9x[2]x[4]^2 - 11x[2]x[3]x[4] - x[2]x[3]^2 + 2x[2]^2*x[4] - x[2]^2*x[3] + 9x[1]x[4]^2 - 11x[1]x[3]x[4] - x[1]x[3]^2 - 11x[1]x[2]x[4] + 15x[1]x[2]x[3] - x[1]x[2]^2 + 2x[1]^2*x[4] - x[1]^2*x[3] - x[1]^2*x[2]
K₄ = -2x[3]^3*x[4] + 3x[2]x[3]^2*x[4] + x[2]x[3]^3 + 3x[2]^2*x[3]x[4] - 4x[2]^2*x[3]^2 - 2x[2]^3*x[4] + x[2]^3*x[3] + 3x[1]x[3]^2*x[4] + x[1]x[3]^3 - 12x[1]x[2]x[3]x[4] + 2x[1]x[2]x[3]^2 + 3x[1]x[2]^2*x[4] + 2x[1]x[2]^2*x[3] + x[1]x[2]^3 + 3x[1]^2*x[3]x[4] - 4x[1]^2*x[3]^2 + 3x[1]^2*x[2]x[4] + 2x[1]^2*x[2]x[3] - 4x[1]^2*x[2]^2 - 2x[1]^3*x[4] + x[1]^3*x[3] + x[1]^3*x[2]
L₄ = (x[1]-x[4])*(x[2]-x[4]) + (x[1]-x[4])*(x[3]-x[4]) + (x[2]-x[4])*(x[3]-x[4])

(x4-x1 + x4-x2 + x4-x3)
(x1-x2)^2 + (x1-x3)^2 + (x2-x3)^2 
6x4^3 - 6x3x4^2 - 6x3^2*x4 - 2x3^3 - 6x2x4^2 + 12x2x3x4 + 6x2x3^2 - 6x2^2*x4 + 6x2^2*x3 - 2x2^3 - 6x1x4^2 + 12x1x3x4 + 6x1x3^2 + 12x1x2x4- 36x1x2x3 + 6x1x2^2 - 6x1^2*x4 + 6x1^2*x3 + 6x1^2*x2 - 2x1^3

p⁴ = p³ # [1 2 3 4]
p³⁺¹₁ = p²⁺¹₁ # [1 3 4; 2]      [1 3 4; 2]
p³⁺¹₂ = p²⁺¹₂ # [1 2 4; 3]      [1 3 4; 2]
p³⁺¹₃ = A₄ # [1 2 3; 4]
p²⁺²₁ = p²⁺¹₁*A₄ - p²⁺¹₁₂ # [1 3; 2 4]
p²⁺²₂ = p²⁺¹₂*A₄ + p²⁺¹₂₂ # [1 2; 3 4]
p²⁺¹⁺¹₁ = p²⁺¹₁*B₄ - p²⁺¹₁₂*A₄ # [1 3; 2; 4]
p²⁺¹⁺¹₂ = p²⁺¹₂*B₄ + p²⁺¹₂₂*A₄ # [1 2; 3; 4]
p²⁺¹⁺¹₃ = p¹⁺¹⁺¹ # [1 4 ; 2; 3]
p¹⁺¹⁺¹⁺¹ = Δ⁴ = p¹⁺¹⁺¹ * H₄ # [1; 2; 3; 4]

p³⁺¹₁₂ = A₄*p²⁺¹₁ + 2p²⁺¹₁₂ # [1 3 4; 2]    [1 2 4; 3]
p³⁺¹₂₂ = A₄*p²⁺¹₂ - 2p²⁺¹₂₂ # [1 2 4; 3]    [1 2 4; 3]
p³⁺¹₃₂ = G₄ # [1 2 3; 4]                [1 2 4; 3]
p²⁺²₁₂ = p²⁺¹₁*C₄ + p²⁺¹₁₂*D₄ # [1 3; 2 4]
p²⁺²₂₂ = p²⁺¹₂*C₄ - p²⁺¹₂₂*D₄ # [1 2; 3 4]
p²⁺¹⁺¹₁₂ = p²⁺¹₁*F₄ + E₄*p²⁺¹₁₂ # [1 3; 2; 4]
p²⁺¹⁺¹₂₂ = p²⁺¹₂*F₄ - E₄*p²⁺¹₂₂ # [1 2 ; 3; 4]
p²⁺¹⁺¹₃₂ = p¹⁺¹⁺¹*A₄ # [1; 2; 3; 4]

p³⁺¹₁₃ = p²⁺¹₁*B₄ + 5p²⁺¹₁₂*A₄ # [1 3 4; 2]     [1 2 3; 4]
p³⁺¹₂₃ = p²⁺¹₂*B₄ - 5p²⁺¹₂₂*A₄ # [1 2 4; 3]     [1 2 3; 4]
p³⁺¹₃₃ = I₄ # [1 2 3; 4]     [1 2 3; 4]
p²⁺¹⁺¹₁₃ = K₄*p²⁺¹₁ + J₄*p²⁺¹₁₂ # [1 3; 2; 4]
p²⁺¹⁺¹₂₃ = K₄*p²⁺¹₂ - J₄*p²⁺¹₂₂ # [1 2 ; 3; 4]
p²⁺¹⁺¹₃₃ = L₄*p¹⁺¹⁺¹ # [1; 2; 3; 4]

# n = 5
A₅ = sum( x[1:4] .- x[5])
B₅  =  (x[1]-x[2])^2 + (x[1]-x[3])^2 + (x[1]-x[4])^2 + (x[2]-x[3])^2 + (x[2]-x[4])^2 + (x[3]-x[4])^2 - A₅^2
G₅ = -12x[5]^2 + 6x[4]x[5] + 3x[4]^2 + 6x[3]x[5] - 4x[3]x[4] + 3x[3]^2 + 6x[2]x[5] - 4x[2]x[4] - 4x[2]x[3] + 3x[2]^2 + 6x[1]x[5] - 4x[1]x[4] - 4x[1]x[3] - 4x[1]x[2] + 3x[1]^2
H₅ = prod(x[5] .- x[1:4])
L₅ = (x[1]-x[5])*(x[2]-x[5]) + (x[1]-x[5])*(x[3]-x[5]) + (x[1]-x[5])*(x[4]-x[5]) + (x[2]-x[5])*(x[3]-x[5]) + (x[2]-x[5])*(x[4]-x[5]) + (x[3]-x[5])*(x[4]-x[5])



p⁵ = p⁴ # [1 2 3 4 5]
p⁴⁺¹₁ = p³⁺¹₁ # [1 3 4 5; 2]
p⁴⁺¹₂ = p³⁺¹₂ # [1 2 4 5; 3]
p⁴⁺¹₃ = p³⁺¹₃ # [1 2 3 5; 4]
p⁴⁺¹₄ = A₅  # [1 2 3 4; 5]
p³⁺²₁ = p²⁺²₁ # [1 3 5; 2 4]
p³⁺²₂ = p²⁺²₂ # [1 2 5; 3 4]
p³⁺²₃ = -3p³⁺¹₁*A₅ + p³⁺¹₁₂  # [1 3 4; 2 5]
p³⁺²₄ = -3p³⁺¹₂*A₅ + p³⁺¹₂₂ # [1 2 4; 3 5]
p³⁺²₅ = p³⁺¹₃*A₅ - p³⁺¹₃₂ # [1 2 3; 4 5]
p³⁺¹⁺¹₁ = 3p³⁺¹₁*B₅ - 2p³⁺¹₁₂*A₅ # [1 3 4; 2; 5]
p³⁺¹⁺¹₂ = 3p³⁺¹₂*B₅ - 2p³⁺¹₂₂*A₅ # [1 2 4; 3; 5]
p³⁺¹⁺¹₃ = p³⁺¹₃*B₅ - 2p³⁺¹₃₂*A₅ # [1 2 3; 4; 5]
p³⁺¹⁺¹₄ = p²⁺¹⁺¹₁₂*A₅ + p²⁺¹⁺¹₁₃ # [1 3 5; 2; 4]
p³⁺¹⁺¹₅ = p²⁺¹⁺¹₂₂*A₅ + p²⁺¹⁺¹₂₃ # [1 4 5; 2; 3]
p²⁺¹⁺¹⁺¹₄ = p¹⁺¹⁺¹⁺¹ # [1 5; 2; 3; 4]
p¹⁺¹⁺¹⁺¹⁺¹ = Δ⁵ = p¹⁺¹⁺¹⁺¹ * H₅ # [1; 2; 3; 4; 5]

p⁴⁺¹₁₂ = 6A₅*p³⁺¹₁ + 10p³⁺¹₁₂ # [1 3 4 5; 2]
p⁴⁺¹₂₂ = 6A₅*p³⁺¹₂ + 10p³⁺¹₂₂ # [1 2 4 5; 3]
p⁴⁺¹₄₂ = G₅
p³⁺¹⁺¹₁₂ = polydiff(Δ⁵,p³⁺¹⁺¹₅)
p³⁺¹⁺¹₅₂ = polydiff(Δ⁵,p³⁺¹⁺¹₁₂)
p³⁺¹⁺¹₁₃ = polydiff(Δ⁵,p³⁺¹⁺¹₅₂)
p³⁺¹⁺¹₅₃ = polydiff(Δ⁵,p³⁺¹⁺¹₁₃)
p³⁺¹⁺¹₁₄ = polydiff(Δ⁵,p³⁺¹⁺¹₅₃)


polys = [p³⁺¹⁺¹₁₂,p³⁺¹⁺¹₁₃,p³⁺¹⁺¹₁₄]
polys = [p³⁺¹⁺¹₅₂,p³⁺¹⁺¹₅₃]

# union of all monomials appearing in any of the polynomials
X = sort!(union(monomials.(polys)...))

# coefficient matrix: rows = monomials, columns = polynomials
A = [coefficient(p, m) for m in X, p in polys]

@test rank(A) == length(polys)   # true iff linearly independent

p²⁺¹⁺¹⁺¹₄₂ = p¹⁺¹⁺¹⁺¹*A₅ # [1 5; 2; 3; 4]
p²⁺¹⁺¹⁺¹₄₃ = p¹⁺¹⁺¹⁺¹*L₅ # [1 5; 2; 3; 4]

p⁴⁺¹₁₃ = 3*p³⁺¹₁*B₅ + 6 * p³⁺¹₁₂*A₅ + 4p³⁺¹₁₃ # [1 3 4 5; 2]
p⁴⁺¹₂₃ = 3*p³⁺¹₂*B₅ + 6 * p³⁺¹₂₂*A₅ + 4p³⁺¹₂₃ # [1 2 4 5; 3]
p⁴⁺¹₄₃ = polydiff(Δ⁵,p²⁺¹⁺¹⁺¹₄₂)

p⁴⁺¹₄₄ = polydiff(Δ⁵,p²⁺¹⁺¹⁺¹₄)

A₆ = sum( x[1:5] .- x[6])
G₆ = (-5x[6]^2 + sum(x[1:5].^2)) + 2*x[6]sum(x[1:5]) - (sum(x[k]sum(x[1:k-1]) for k=1:5))
H₆ = prod(x[6] .- x[1:5])

p⁶ = p⁵ # [1 2 3 4 5 6]
p⁵⁺¹₁ = p⁴⁺¹₁ # [1 3 4 5 6; 2]
p⁵⁺¹₂ = p⁴⁺¹₂ # [1 2 4 5 6; 3]
p⁵⁺¹₃ = p⁴⁺¹₃ # [1 2 3 5 6; 4]
p⁵⁺¹₄ = p⁴⁺¹₄ # [1 2 3 4 6; 5]
p⁵⁺¹₅ = A₆  # [1 2 3 4 5; 6]

p⁵⁺¹₁₂ = 4A₆*p⁴⁺¹₁ + p⁴⁺¹₁₂ # [1 3 4 5 6; 2]
p⁵⁺¹₂₂ = 4A₆*p⁴⁺¹₂ + p⁴⁺¹₂₂ # [1 2 4 5 6; 3]
p⁵⁺¹₅₂ = G₆ # [1 2 3 4 5; 6]

p¹⁺¹⁺¹⁺¹⁺¹⁺¹ = Δ⁶ = p¹⁺¹⁺¹⁺¹⁺¹ * H₆ # [1; 2; 3; 4; 5]


####
# content vectors
####
@testset "content vector" begin
    @test contentvectorequals(p², x, [1])
    @test contentvectorequals(p¹⁺¹, x, [-1])
    @test contentvectorequals(p³, x, [1,2])
    @test contentvectorequals(p²⁺¹₁, x, [-1,1])
    @test contentvectorequals(p²⁺¹₂, x, [1,-1])
    @test contentvectorequals(p²⁺¹₁₂, x, [-1,1])
    @test contentvectorequals(p²⁺¹₂₂, x, [1,-1])
    @test contentvectorequals(p¹⁺¹⁺¹, x, [-1,-2])

    @test contentvectorequals(p⁴, x, [1,2,3])

    @test contentvectorequals(p³⁺¹₁, x, [-1,1,2])
    @test contentvectorequals(p³⁺¹₂, x, [1,-1,2])
    @test contentvectorequals(p³⁺¹₃, x, [1,2,-1])
    @test contentvectorequals(p³⁺¹₁₂, x, [-1,1,2])
    @test contentvectorequals(p³⁺¹₂₂, x, [1,-1,2])
    @test contentvectorequals(p³⁺¹₃₂, x, [1,2,-1])
    @test contentvectorequals(p³⁺¹₁₃, x, [-1,1,2])
    @test contentvectorequals(p³⁺¹₂₃, x, [1,-1,2])
    @test contentvectorequals(p³⁺¹₃₃, x, [1,2,-1])

    @test contentvectorequals(p²⁺²₁, x, [-1,1,0])
    @test contentvectorequals(p²⁺²₂, x, [1,-1,0])
    @test contentvectorequals(p²⁺²₁₂, x, [-1,1,0])
    @test contentvectorequals(p²⁺²₂₂, x, [1,-1,0])

    @test contentvectorequals(p²⁺¹⁺¹₁, x, [-1,1,-2])
    @test contentvectorequals(p²⁺¹⁺¹₂, x, [1,-1,-2])
    @test contentvectorequals(p²⁺¹⁺¹₃, x, [-1,-2,1])
    @test contentvectorequals(p²⁺¹⁺¹₁₂, x, [-1,1,-2])
    @test contentvectorequals(p²⁺¹⁺¹₂₂, x, [1,-1,-2])
    @test contentvectorequals(p²⁺¹⁺¹₃₂, x, [-1,-2,1])
    @test contentvectorequals(p²⁺¹⁺¹₁₃, x, [-1,1,-2])
    @test contentvectorequals(p²⁺¹⁺¹₂₃, x, [1,-1,-2])
    @test contentvectorequals(p²⁺¹⁺¹₃₃, x, [-1,-2,1])

    @test contentvectorequals(p¹⁺¹⁺¹⁺¹, x, [-1,-2,-3])

    @test contentvectorequals(p⁵, x, [1,2,3,4])

    @test contentvectorequals(p⁴⁺¹₁, x, [-1,1,2,3])
    @test contentvectorequals(p⁴⁺¹₂, x, [1,-1,2,3])
    @test contentvectorequals(p⁴⁺¹₃, x, [1,2,-1,3])
    @test contentvectorequals(p⁴⁺¹₄, x, [1,2,3,-1])
    @test contentvectorequals(p⁴⁺¹₁₂, x, [-1,1,2,3])
    @test contentvectorequals(p⁴⁺¹₂₂, x, [1,-1,2,3])
    @test contentvectorequals(p⁴⁺¹₄₂, x, [1,2,3,-1])
    @test contentvectorequals(p⁴⁺¹₁₃, x, [-1,1,2,3])
    @test contentvectorequals(p⁴⁺¹₂₃, x, [1,-1,2,3])

    @test contentvectorequals(p³⁺²₁, x, [-1,1,0,2])
    @test contentvectorequals(p³⁺²₂, x, [1,-1,0,2])
    @test contentvectorequals(p³⁺²₃, x, [-1,1,2,0])
    @test contentvectorequals(p³⁺²₄, x, [1,-1,2,0])
    @test contentvectorequals(p³⁺²₅, x, [1,2,-1,0])

    @test contentvectorequals(p³⁺¹⁺¹₁, x, [-1,1,2,-2])
    @test contentvectorequals(p³⁺¹⁺¹₂, x, [1,-1,2,-2])
    @test contentvectorequals(p³⁺¹⁺¹₃, x, [1,2,-1,-2])
    @test contentvectorequals(p³⁺¹⁺¹₄, x, [-1,1,-2,2])
    @test contentvectorequals(p³⁺¹⁺¹₅, x, [1,-1,-2,2])
    @test contentvectorequals(p³⁺¹⁺¹₁₂, x, [-1,1,2,-2]) 
    @test contentvectorequals(p³⁺¹⁺¹₅₂, x, [1,-1,-2,2])
    @test contentvectorequals(p³⁺¹⁺¹₁₃, x, [-1,1,2,-2]) 
    @test contentvectorequals(p³⁺¹⁺¹₅₃, x, [1,-1,-2,2])

    @test contentvectorequals(p²⁺¹⁺¹⁺¹₄, x, [-1,-2,-3,1])
    @test contentvectorequals(p²⁺¹⁺¹⁺¹₄₂, x, [-1,-2,-3,1])
    @test contentvectorequals(p²⁺¹⁺¹⁺¹₄₃, x, [-1,-2,-3,1])

    @test contentvectorequals(p¹⁺¹⁺¹⁺¹⁺¹, x, [-1,-2,-3,-4])

    @test contentvectorequals(p⁶, x, [1,2,3,4,5])

    @test contentvectorequals(p⁵⁺¹₁, x, [-1,1,2,3,4])
    @test contentvectorequals(p⁵⁺¹₂, x, [1,-1,2,3,4])
    @test contentvectorequals(p⁵⁺¹₃, x, [1,2,-1,3,4])
    @test contentvectorequals(p⁵⁺¹₄, x, [1,2,3,-1,4])
    @test contentvectorequals(p⁵⁺¹₅, x, [1,2,3,4,-1])
    @test contentvectorequals(p⁵⁺¹₁₂, x, [-1,1,2,3,4])
    @test contentvectorequals(p⁵⁺¹₂₂, x, [1,-1,2,3,4])
    @test contentvectorequals(p⁵⁺¹₅₂, x, [1,2,3,4,-1])
end


#####
# PDEs
####

@testset "laplacian" begin
    @test all(iszero, [laplacian(p², x[1:2], k) for k=1:2])
    @test all(iszero, [laplacian(p¹⁺¹, x[1:2], k) for k=1:2])
    @test all(iszero, [laplacian(p³, x[1:3], k) for k=1:3])
    for p in (p²⁺¹₁, p²⁺¹₂, p²⁺¹₁₂, p²⁺¹₂₂)
        @test all(iszero, [laplacian(p, x[1:3], k) for k=1:3])
    end        

    for p in (p¹⁺¹⁺¹,)
        @test all(iszero, [laplacian(p, x[1:3], k) for k=1:3])
    end        

    for p in (p⁴,)
        @test all(iszero, [laplacian(p, x[1:4], k) for k=1:4])
    end

    for p in (p³⁺¹₁,p³⁺¹₂,p³⁺¹₃,p³⁺¹₁₂,p³⁺¹₂₂,p³⁺¹₃₂,p³⁺¹₁₃,p³⁺¹₂₃,p³⁺¹₃₃)
        @test all(iszero, [laplacian(p, x[1:4], k) for k=1:4])
    end    

    for p in (p²⁺²₁,p²⁺²₂,p²⁺²₁₂,p²⁺²₂₂)
        @test all(iszero, [laplacian(p, x[1:4], k) for k=1:4])
    end

    for p in (p²⁺¹⁺¹₁,p²⁺¹⁺¹₂,p²⁺¹⁺¹₃,p²⁺¹⁺¹₁₂,p²⁺¹⁺¹₂₂,p²⁺¹⁺¹₃₂,p²⁺¹⁺¹₁₃,p²⁺¹⁺¹₂₃,p²⁺¹⁺¹₃₃)
        @test all(iszero, [laplacian(p, x[1:4], k) for k=1:4])
    end

    for p in (p⁵,p⁴⁺¹₁,p⁴⁺¹₂,p⁴⁺¹₃,p⁴⁺¹₄,p³⁺²₁,p³⁺²₂,p³⁺²₃,p³⁺²₄,p³⁺²₅,p³⁺¹⁺¹₁,p³⁺¹⁺¹₂,p³⁺¹⁺¹₃,p²⁺¹⁺¹⁺¹₄,p¹⁺¹⁺¹⁺¹⁺¹,p⁴⁺¹₁₂,p⁴⁺¹₂₂,p⁴⁺¹₄₂,p²⁺¹⁺¹⁺¹₄₂,p⁴⁺¹₁₃,p⁴⁺¹₂₃,p⁴⁺¹₄₃,p²⁺¹⁺¹⁺¹₄₃,p⁴⁺¹₄₄)
        @test all(iszero, [laplacian(p, x[1:5], k) for k=1:5])
    end
end


#######
# duality
#######

# p²⁺¹₁(D)*Δ³
@test polydiff(Δ³, p²⁺¹₁) == differentiate(Δ³, x[2]) - differentiate(Δ³, x[1]) == p²⁺¹₂₂ == B₃
# p²⁺¹₂(D)*Δ
@test polydiff(Δ³, p²⁺¹₂) == -(2differentiate(Δ³, x[3]) - differentiate(Δ³, x[1]) - differentiate(Δ³, x[2])) == 3p²⁺¹₁₂


@test polydiff(Δ⁴, p¹⁺¹⁺¹⁺¹) == 288p⁴

@test 9polydiff(Δ⁴, p³⁺¹₁) == p²⁺¹⁺¹₂₃
@test -3polydiff(Δ⁴, p³⁺¹₂) == p²⁺¹⁺¹₁₃
@test polydiff(Δ⁴, p³⁺¹₃) == 4p²⁺¹⁺¹₃₃

@test -3polydiff(Δ⁴, p³⁺¹₁₂) == 4p²⁺¹⁺¹₂₂
@test polydiff(Δ⁴, p³⁺¹₂₂) == 4p²⁺¹⁺¹₁₂
@test polydiff(Δ⁴, p³⁺¹₃₂) == 16p²⁺¹⁺¹₃₂

@test polydiff(Δ⁴, p³⁺¹₁₃) == 40p²⁺¹⁺¹₂
@test polydiff(Δ⁴, p³⁺¹₂₃) == -120p²⁺¹⁺¹₁
@test polydiff(Δ⁴, p³⁺¹₃₃) == 480p²⁺¹⁺¹₃

@test polydiff(Δ⁴, p²⁺²₁) == 2p²⁺²₂₂
@test polydiff(Δ⁴, p²⁺²₂) == -6p²⁺²₁₂

@test polydiff(Δ⁴, p²⁺¹⁺¹₁) == -8p³⁺¹₂₃
@test polydiff(Δ⁴, p²⁺¹⁺¹₂) == 24p³⁺¹₁₃
@test polydiff(Δ⁴, p²⁺¹⁺¹₃) == 2p³⁺¹₃₃

@test polydiff(Δ⁴, p²⁺¹⁺¹₁₂) == 192p³⁺¹₂₂
@test polydiff(Δ⁴, p²⁺¹⁺¹₂₂) == -576p³⁺¹₁₂
@test polydiff(Δ⁴, p²⁺¹⁺¹₃₂) == 48p³⁺¹₃₂

@test polydiff(Δ⁴, p²⁺¹⁺¹₁₃) == -1728p³⁺¹₂
@test polydiff(Δ⁴, p²⁺¹⁺¹₂₃) == 9*576p³⁺¹₁
@test polydiff(Δ⁴, p²⁺¹⁺¹₃₃) == -3*48p³⁺¹₃


@test polydiff(Δ⁴, p⁴) == p¹⁺¹⁺¹⁺¹

@test polydiff(Δ⁵, p¹⁺¹⁺¹⁺¹⁺¹) == 34560p⁵

@test polydiff(Δ⁵,p²⁺¹⁺¹⁺¹₄) == p⁴⁺¹₄₄
@test polydiff(Δ⁵,p²⁺¹⁺¹⁺¹₄₂) == p⁴⁺¹₄₃
@test polydiff(Δ⁵,p²⁺¹⁺¹⁺¹₄₃) == -2880p⁴⁺¹₄₂

@test polydiff(Δ⁵,p³⁺¹⁺¹₅) == p³⁺¹⁺¹₁₂


#######
# experiments
########
# quadratic form for higher-order
@test p²⁺¹₁₂ == x[1:3]'*[-1 0 1; 0 1 -1; 1 -1 0]*x[1:3]
@test p²⁺¹₂₂ == x[1:3]'*[-1 2 -1; 2 -1 -1; -1 -1 2]*x[1:3]




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




# 4 = 2+1+1







# (x[4]-x[2])*(x[4]-x[1])*(x[2]-x[1])
# (x[4]-x[3])*(x[4]-x[1])*(x[3]-x[1])


x[4]-x[1] + x[4]-x[2] + x[4]-x[3] # [1 2; 3; 4]



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

# [1 3 4 5; 2]
p³⁺¹₁

# [1 2 3 4; 5]
p⁴⁺¹₁ = sum(x[5]-x[k] for k = 1:4)


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


####
# simplify b
###

@test (x[1]-x[2])^2  - A₃^2 == -4C₃
@test (x[1]-x[2])^2 + (x[1]-x[3])^2 + (x[2]-x[3])^2 - A₄^2 == B₄


p²⁺²⁺¹₁ = p²⁺²₁
@polyvar α[1:2]
p²⁺¹⁺¹⁺¹₁ = α[1]p²⁺¹⁺¹₁*B₅ + α[2]A₅*p²⁺¹⁺¹₁₂

[-1,1,2,3]
@test gelfand(G₅, x, 1) == G₅
@test gelfand(G₅, x, 2) == 2G₅
@test gelfand(G₅, x, 3) == 3G₅
@test gelfand(G₅, x, 4) == -G₅

err = gelfand(G₅, x, 4) + G₅
t = x[4]x[5]
coefficient(err, t)
[coefficient(err, t*α[k]) for k=1:3]

# 22α[1] == 7α[2] + 6α[3]


# commented out because it didn't satisfy Laplace
G₅ = x[1]^2+x[2]^2+x[3]^2+x[4]^2 - 4x[5]^2 + (3*(x[1]+x[2]+x[3]+x[4])x[5] + 2 * (  - x[2]x[3] - x[1]x[3] - x[1]x[2] - x[1]x[4] - x[2]x[4] - x[3]x[4]))
G₅ = -12x[5]^2 + 6x[4]x[5] + 3x[4]^2 + 6x[3]x[5] - 4x[3]x[4] + 3x[3]^2 + 6x[2]x[5] - 4x[2]x[4] - 4x[2]x[3] + 3x[2]^2 + 6x[1]x[5] - 4x[1]x[4] - 4x[1]x[3] - 4x[1]x[2] + 3x[1]^2

p⁵\

p⁴⁺¹₄₂ = 5G₅+(A₅)*(x[1]+x[2]+x[3]+x[4]+x[5])
@test contentvectorequals(p⁴⁺¹₄₂, x, [1,2,3,-1])
@test all(iszero, [laplacian(p⁴⁺¹₄₂, x[1:5], k) for k=1:5])
laplacian(5G₅+(A₅)*(x[1]+x[2]+x[3]+x[4]+x[5]), x[1:5], 1) == -5A₅
laplacian((A₅)*(x[1]+x[2]+x[3]+x[4]+x[5]),x,1) == 5(A₅)

x[1]^2+x[2]^2+x[3]^2 - 3x[4]^2 + 2(x[3]x[4] + x[2]x[4] - x[2]x[3] + x[1]x[4] - x[1]x[3] - x[1]x[2])
B₃
@test -B₃ == (-2x[3]^2 + sum(x[1:2].^2)) + 2x[3]sum(x[1:2]) - 4*(sum(x[k]sum(x[1:k-1]) for k=1:2))
@test G₄ == (-3x[4]^2 + sum(x[1:3].^2)) + 2x[4]sum(x[1:3]) - 2*(sum(x[k]sum(x[1:k-1]) for k=1:3))
@test G₅/3 == (-4x[5]^2 + sum(x[1:4].^2)) + 2*x[5]sum(x[1:4]) - 4/3*(sum(x[k]sum(x[1:k-1]) for k=1:4))



@test I₄ == 6x[4]^3 - 6x[4]^2 * (x[1]+x[2]+x[3]) - 6x[4] * (x[3]^2 + x[2]^2 + x[1]^2 - 2*(x[1]x[3]+x[1]x[2]+x[2]x[3])) - 2*(x[1]^3 + x[2]^3 + x[3]^3 - 3 * (x[1]x[2]^2 + x[1]x[3]^2 + x[2]x[3]^2 + x[1]^2*x[2] + x[1]^2*x[3] + x[2]^2*x[3]) + 18*x[1]x[2]x[3])


a = 180
b = -120


c = b - 6 * a
@polyvar a b c
f = a*p²⁺¹⁺¹₁*B₅ + b*p²⁺¹⁺¹₁₂*A₅ + c*p²⁺¹⁺¹₁₃
@test gelfand(f, x, 1) == -f
@test gelfand(f, x, 2) == f
@test gelfand(f, x, 3) == -2f


err = gelfand(f, x, 4) - (2f)
t = x[2]x[4]^2*x[5]^2
coefficient(err, t*a)
coefficient(err, t*b)
coefficient(err, t*c)


laplacian(f,x,2)

gelfand(p³⁺¹⁺¹₅,x, 4) - 2p³⁺¹⁺¹₅




######
# diagonalise?
######

ρ = Representation(2,1)

@polyvar c
𝐩 = [p²⁺¹₁, -1/sqrt(3) * p²⁺¹₂]

@test subs.(𝐩, x[2]=>x[1], x[1]=>x[2])  == ρ.generators[1]*𝐩
@test all(subs.(𝐩, x[3]=>x[2], x[2]=>x[3])  .≈ ρ.generators[2]*𝐩)

_, Q = blockdiagonalize(ρ ⊗ ρ)


@test (ρ ⊗ ρ).generators[1] * Q[:,1] == -Q[:,1]
@test (ρ ⊗ ρ).generators[2] * Q[:,1] == -Q[:,1]

Q' * (ρ ⊗ ρ).generators[1] *Q
Q' * (ρ ⊗ ρ).generators[2] *Q

@test (ρ ⊗ ρ).generators[1] * Q[:,1] ≈ -Q[:,1]
@test (ρ ⊗ ρ).generators[2] * Q[:,1] ≈ -Q[:,1]

@test (ρ ⊗ ρ).generators[1] * Q[:,4] ≈ Q[:,4]
@test (ρ ⊗ ρ).generators[2] * Q[:,4] ≈ Q[:,4]



@test ρ.generators[1]*reshape(Q[:,1], 2, 2)*ρ.generators[1]' == reshape(Q[:,3], 2, 2)
@test ρ.generators[2]*reshape(Q[:,3], 2, 2)*ρ.generators[2]' == reshape(Q[:,3], 2, 2)


@test contentvectorisapprox(𝐩[1]^2 + 𝐩[2]^2, x, [1,2])

@test contentvectorisapprox(𝐩[1]𝐩[2], x, [-1,1])
@test contentvectorisapprox(𝐩[1]^2 - 𝐩[2]^2, x, [1,-1])

@test p²⁺¹₁₂ ≈ -sqrt(3)𝐩[1]𝐩[2]
@test p²⁺¹₂₂ ≈ -3/2*(𝐩[1]^2 - 𝐩[2]^2)

