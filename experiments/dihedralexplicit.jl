using MultivariateOrthogonalPolynomials, InfiniteArrays, LazyArrays, BlockArrays, DomainSets, StaticArrays, ClassicalOrthogonalPolynomials
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


struct DihedralInvariantPolynomial{T} <: Basis{T} end
DihedralInvariantPolynomial() = DihedralInvariantPolynomial{Float64}()

axes(::DihedralInvariantPolynomial) = (Inclusion(ChebyshevInterval() × ChebyshevInterval()), BlockedOneTo((2:∞) .^ 2 .÷ 4))




function getindex(Q::DihedralInvariantPolynomial, 𝐱::SVector{2}, Kk::BlockIndex)
    x,y = 𝐱
    K,k = block(Kk), blockindex(Kk)
    ℓ = 2*(Int(K)-1)
    μ = 2*(k-1)
    legendrep(ℓ,x)legendrep(μ,y)+legendrep(μ,x)legendrep(ℓ,y)
end

getindex(Q::DihedralInvariantPolynomial, 𝐱::SVector{2}, k::Int) = Q[𝐱,findblockindex(axes(Q,2),k)]

Q = DihedralInvariantPolynomial()
