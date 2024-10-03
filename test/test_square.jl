using ClassicalOrthogonalPolynomials, LazyBandedMatrices, LinearAlgebra, BlockArrays, Test

## Weak Laplacian

W = Weighted(Jacobi(1,1))
x = axes(W,1)
D = Derivative(x)
D² = -((D*W)'*(D*W))
M = W'W
Δ = KronTrav(D²,M) + KronTrav(M,D²)

M2 = KronTrav(M,M)

K = Block.(1:5)
Δ[K,K]

## Note it decomposes into 4 independent sub-matrices, i.e. it can be trivially parallelised over 4 cores. This is because the Tensor product basis is already
# decomposed into irreps of ℤ_2^2 corresponding to reflection symmetries (x,y) -> (x,-y) and (x,y) -> (-x,y)


## D_8 group https://groupprops.subwiki.org/wiki/Linear_representation_theory_of_dihedral_group:D8
#  with  generators x, a has following irreps: 
#  x             a
#  1             1              (trivial)
#  1            -1
# -1             1
# -1            -1              (sign)
# [1 0; 0 -1]   [0 -1; 1 0]      (faithful)
#
# The tensor product basis of degree n
#
#   [W_n(x) W_0(y), W_{n-1}(x) W_1(y), …, W_0(x) W_n(y)]
#
# induces the representation by applying the actions (x,y) -> (x,-y) and (x,y) -> (-y,x)
#   Degree 0:       1                      1                (i.e. trivial)
#   Degree 1:       Diagonal([1,-1])       [0 -1; 1 0]       (i.e. faithful)
#   Degree 2:       Diagonal([1,-1,1])     [0 0 1; 0 1 0; 1 0 0]
#   Degree 3:       Diagonal([1,-1,1,-1])  [0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0]
#
#   Degree n:       Diagonal((-1).^(0:n))  I(n+1)[end:-1:1,:]

𝕎 = (n,x,y) -> [W[x,n-k+1]*W[y,k+1] for k=0:n]

x,y = 0.1,0.2

n = 0
@test 𝕎(n,x,-y) == 𝕎(n,x,y)
@test 𝕎(n,-y,x) == 𝕎(n,x,y)

n = 1
@test 𝕎(n,x,-y) == Diagonal([1,-1]) * 𝕎(n,x,y)
@test 𝕎(n,-y,x) == [0 -1; 1 0] * 𝕎(n,x,y)

n = 10
@test 𝕎(n,x,-y) == Diagonal((-1).^(0:n)) * 𝕎(n,x,y)
@test 𝕎(n,-y,x) == Diagonal((-1).^(0:n))[end:-1:1,:] * 𝕎(n,x,y)


# We want to find Q that reduces these to irreps. Degree 0 and 1 are done. We use a different linear combination for other degrees
# inorder that the representation is broken up into irreps:
#
# irreps        1,1                              1,-1                         -1,1                             -1,-1                       [1 0; 0 -1],[0 -1; 1 0]
# Degree 0:     W_0(x)W_0(y)                                    
# Degree 1:                                                                                                                                [W_1(x), W_1(y)]
# Degree 2:     W_2(x)+W_2(y)                    W_2(x)-W_2(y)                                                 W_1(x)W_1(y)
# Degree 3:                                                                                                                                [W_3(x), W_3(y)]
#                                                                                                                                          [W_1(x)W_2(y), W_2(x)W_1(y)]
# Degree 4:     W_4(x)+W_4(y)                    W_4(x)-W_4(y)                 W_3(x)W_1(y)-W_1(x)W_3(y)       W_3(x)W_1(y)+W_1(x)W_3(y)       
#               W_2(x)W_2(y)                            
# Degree 5:                                                                                                                                [W_5(x), W_5(y)]
#                                                                                                                                          [W_1(x)W_4(y), W_4(x)W_1(y)]
#                                                                                                                                          [W_3(x)W_2(y), W_2(x)W_3(y)]
# Degree 6:     W_6(x)+W_6(y)                    W_6(x)-W_6(y)                 W_5(x)W_1(y)-W_1(x)W_5(y)        W_5(x)W_1(y)+W_1(x)W_5(y)       
#               W_4(x)W_2(y)+W_2(x)W_4(y)        W_4(x)W_2(y)-W_2(x)W_4(y)                                      W_3(x)W_3(y)

# This corresponds to the following orthogonal transformations:

μ = 1/sqrt(2)
Q = mortar(Diagonal([Matrix(I,1,1),
Matrix(I,2,2),
[μ 0 μ; μ 0 -μ; 0 1 0],
[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0],
[μ 0 0 0 μ; 0 0 1 0 0; μ 0 0 0 -μ; 0 μ 0  -μ 0; 0 μ 0 μ 0],
[1 0 0 0 0 0; 0 0 0 0 0 1; 0 0 1 0 0 0; 0 0 0 1 0 0; 0 0 0 0 1 0; 0 1 0 0 0 0],
[μ 0 0 0 0 0 μ; 0 0 μ 0 μ 0 0; μ 0 0 0 0 0 -μ; 0 0 μ 0  -μ 0 0; 0 μ 0 0 0 -μ 0; 0 μ 0 0 0 μ 0; 0 0 0 1 0 0 0]]))

@test Q'Q ≈ I


𝕍 = (n,x,y) -> Q[Block(n+1,n+1)] * 𝕎(n,x,y)
n = 0
# trivial
@test 𝕍(n,x,-y) ≈ 𝕍(n,x,y)
@test 𝕍(n,-y,x) ≈ 𝕍(n,x,y)

n = 1
# faithful
@test 𝕍(n,x,-y) ≈ [1 0; 0 -1]*𝕍(n,x,y)
@test 𝕍(n,-y,x) ≈ [0 -1; 1 0]*𝕍(n,x,y)

n = 2
# trival, (1,-1), sign
@test 𝕍(n,x,-y) ≈ Diagonal([1,1,-1])*𝕍(n,x,y)
@test 𝕍(n,-y,x) ≈ Diagonal([1,-1,-1])*𝕍(n,x,y)

n = 3
# faithful (2x)
@test 𝕍(n,x,-y) ≈ blockdiag(fill(sparse([1 0; 0 -1]),2)...)*𝕍(n,x,y)
@test 𝕍(n,-y,x) ≈ blockdiag(fill(sparse([0 -1; 1 0]),2)...)*𝕍(n,x,y)

n = 4
# trivial (2x), (1,-1), (-1,1), sign
@test 𝕍(n,x,-y) ≈ Diagonal([1,1,1,-1,-1])*𝕍(n,x,y)
@test 𝕍(n,-y,x) ≈ Diagonal([1,1,-1,1,-1])*𝕍(n,x,y)

n = 5
# faithful (3x)
@test 𝕍(n,x,-y) ≈ blockdiag(fill(sparse([1 0; 0 -1]),3)...)*𝕍(n,x,y)
@test 𝕍(n,-y,x) ≈ blockdiag(fill(sparse([0 -1; 1 0]),3)...)*𝕍(n,x,y)

n = 6
# trivial (2x), (1,-1) (2x), (-1,1), sign (2x)
@test 𝕍(n,x,-y) ≈ Diagonal([1,1,1,1,-1,-1,-1])*𝕍(n,x,y)
@test 𝕍(n,-y,x) ≈ Diagonal([1,1,-1,-1,1,-1,-1])*𝕍(n,x,y)


# Q conjugated with the Laplacian breaks up in irreps

K = Block.(1:7)

Q* Δ[K,K] * Q'


# We even get an extra parallelisation because for the faithful the the x and y don't communicate!
# So we can trivially parallelise onto 6 cores

kr = [1,4,11,12,22,23,
2,7,9,16,18,20,
3,8,10,17,19,21,
5,13,24,25,
6,14,26,27,
15,28]

(Q* Δ[K,K] * Q')[kr,kr]

(Q* (Δ[K,K] + M2[K,K]) * Q')[kr,kr]



###
# general case for Q
n = 6

basisQ(n) = basisQ(Float64, n)
function basisQ(T, n)
    Q = zeros(T, n, n)
    if iseven(n)
        for k = 1:2:n÷2
            Q[k,k] = 1
            Q[k+1,n-k+1] = 1
        end
        m = last(1:2:n÷2)+1
        for k = 1:2:n÷2-1
            Q[k+m+1,k+1] = 1
            Q[k+m,n-k] = 1
        end
    elseif mod(n,4) == 1
        for k = 1:n÷4
            Q[k,2k-1] = Q[k,n-2k+2] = 1/sqrt(T(2))
        end
        m = n÷4
        Q[m+1,n÷2+1] = 1
        m = n÷4+1
        for k = 1:n÷4
            Q[k+m,2k-1] = 1/sqrt(T(2))
            Q[k+m,n-2k+2] = -1/sqrt(T(2))
        end
        m = n÷2+1
        for k = 1:n÷4
            Q[k+m,2k] = 1/sqrt(T(2))
            Q[k+m,n-2k+1] = -1/sqrt(T(2))
        end
        m = 3*(n÷4)+1
        for k = 1:n÷4
            Q[k+m,2k] = 1/sqrt(T(2))
            Q[k+m,n-2k+1] = 1/sqrt(T(2))
        end
    elseif mod(n,4) == 3
        for k = 1:n÷4+1
            Q[k,2k-1] = Q[k,n-2k+2] = 1/sqrt(T(2))
            m = n÷4+1
            Q[k+m,2k-1] = 1/sqrt(T(2))
            Q[k+m,n-2k+2] = -1/sqrt(T(2))
        end
        m = n÷2+1
        for k = 1:n÷4
            Q[k+m,2k] = 1/sqrt(T(2))
            Q[k+m,n-2k+1] = -1/sqrt(T(2))
        end
        m = 3*(n÷4)+2
        for k = 1:n÷4
            Q[k+m,2k] = 1/sqrt(T(2))
            Q[k+m,n-2k+1] = 1/sqrt(T(2))
        end
        Q[end,n÷2+1] = 1
    end
    Q
end

for k = 1:7
    @test basisQ(k) ≈ Q[Block(k,k)]
end
