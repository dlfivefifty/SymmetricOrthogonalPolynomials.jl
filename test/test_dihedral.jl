####
# we try to understand dihedral-adapted OPs from their invariant OPs.
#
# Let's start with the scalar irreps. We have a simple basis of dihedral invariant OPs:

using SymmetricOrthogonalPolynomials, ClassicalOrthogonalPolynomials
P = Legendre()
Q = DihedralInvariant(P)



###
# Now consider the Faithful irrep. We have an explicit basis:
###

n = 3
Q = (n,k,x,y) -> [legendrep(n-2k,x)legendrep(2k,y), legendrep(2k,x)legendrep(n-2k,y)]
for n = 1:2:3, m = 1:2:3, k =1:n÷2, j=1:m÷2
    if n ≠ m || k ≠ j
        @test sum(Q(n,k,x,y)'Q(m,j,x,y) for x in -1..1, y in -1..1) ≈ 0 atol=1E-13
    end
end

# We want to express this in terms of invariant polynomials times basic forms. In particular, we have
x,y = 0.1,0.2
@test 3Q(3,0,x,y) + 5Q(3,1,x,y) ≈ [x,y] * (15*(x^2+y^2)-14)/2
@test 7Q(3,0,x,y) - 9Q(3,1,x,y) ≈ [x*(35x^2-27y^2-12), y*(35y^2-27x^2-12)]/2

@test sum((3Q(3,0,x,y) + 5Q(3,1,x,y))'*[x,y] for x in -1..1, y in -1..1) ≈ 0 atol=1E-13
@test sum((7Q(3,0,x,y) - 9Q(3,1,x,y))'*[x,y] for x in -1..1, y in -1..1) ≈ 0 atol=1E-13
@test sum((7Q(3,0,x,y) - 9Q(3,1,x,y))'*(3Q(3,0,x,y) + 5Q(3,1,x,y)) for x in -1..1, y in -1..1) ≈ 0 atol=1E-13
@test sum((7Q(3,0,x,y) - 9Q(3,1,x,y))'*[x,y]*(x^2+y^2) for x in -1..1, y in -1..1) ≈ 0 atol=1E-13


for j = 0:2
    @test sum((7Q(3,0,x,y) - 9Q(3,1,x,y))'*Q(5,j,x,y) for x in -1..1, y in -1..1) ≈ 0 atol=1E-13
end

using DynamicPolynomials
@polyvar x y

x,y = 0.1,0.2
@test 5*8*Q(5,0,x,y) + 9*8Q(5,2,x,y) + 80Q(5,1,x,y) ≈ [x,y]*(315*(x^4+y^4) + 300*x^2*y^2 - 450*(x^2+y^2) + 162) ≈
    (21*2* (x^2+y^2)-156/7.5)*(3Q(3,0,x,y) + 5Q(3,1,x,y)) - (330*x^2*y^2 - 16.4)*Q(1,0,x,y)
