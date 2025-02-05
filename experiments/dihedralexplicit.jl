using MultivariateOrthogonalPolynomials, InfiniteArrays, LazyArrays, BlockArrays, DomainSets, StaticArrays, ClassicalOrthogonalPolynomials, BlockBandedMatrices
using ContinuumArrays: Basis
using BlockArrays: block, blockindex
import Base: axes, getindex

########
# Invariant polynomials with respect to D_4 are given by
# P_{2k}(x) P_{2j}(y) +  P_{2j}(x) P_{2k}(y)
# The first few are 
# 1
# -----
# P_2(x) + P_2(y)
# ----
# P_4(x) + P_4(y)
# 2P_2(x) P_2(y)
# -----
# P_6(x) + P_6(y)
# P_4(x)P_2(y) + P_2(x) P_4(y)
# ----
# P_8(x) + P_8(y)
# P_6(x)P_2(y) + P_2(x)P_6(y)
# 2P_4(x)P_4(y)
########


struct DihedralInvariantLegendre{T} <: Basis{T} end
DihedralInvariantLegendre() = DihedralInvariantLegendre{Float64}()

axes(::DihedralInvariantLegendre) = (Inclusion(ChebyshevInterval() × ChebyshevInterval()), BlockedOneTo((2:∞) .^ 2 .÷ 4))




function getindex(Q::DihedralInvariantLegendre, 𝐱::SVector{2}, Kk::BlockIndex)
    x,y = 𝐱
    K,k = block(Kk), blockindex(Kk)
    ℓ = 2*(Int(K)-1)
    μ = 2*(k-1)
    (legendrep(ℓ,x)legendrep(μ,y)+legendrep(μ,x)legendrep(ℓ,y))/sqrt(2 - (ℓ == μ))
end

getindex(Q::DihedralInvariantLegendre, 𝐱::SVector{2}, k::Int) = Q[𝐱,findblockindex(axes(Q,2),k)]

Q = DihedralInvariantLegendre()
x,y = 0.1,0.2
for K = Block.(1:10)
    @test Q[SVector(x,y),K] == Q[SVector(y,x),K] == Q[SVector(x,-y),K]
end


P² = KronPolynomial(Legendre(), Legendre())

transform(P², splat((x,y) -> exp(x^2+y^2)*cos(x^2 * y^2)))

# conversion from even tensor to invariant
N = 5
R = BlockBandedMatrix{Float64}(undef, (axes(Q,2)[ Block.(Base.oneto(N))], BlockedOneTo(cumsum(1:2:2N))), (0,0)); fill!(R, 0)
for K = 1:2:N
    for k = 1:(K÷2)
        @show K,k
        R[Block(K,K)[k,2k-1]] = R[Block(K,K)[k,2K-2k+2]] = 1/sqrt(2)
    end
end
R[Block(1,1)] .= 1
R[Block(2,2)] = [1/sqrt(2) 0 1/sqrt(2)]
R[Block(3,5)] = 



X = jacobimatrix(Val(1), P²)
Y = jacobimatrix(Val(1), P²)


