########
# Invariant polynomials with respect to S_2 are given by
# P_{k}(x) P_{j}(y) +  P_{j}(x) P_{k}(y)
# The first few are 
# 1
# -----
# P_1(x) + P_1(y)
# ----
# P_2(x) + P_2(y)
# 2P_1(x) P_1(y)
# -----
# P_3(x) + P_3(y)
# P_2(x)P_1(y) + P_1(x) P_2(y)
# ----
# P_4(x) + P_4(y)
# P_3(x)P_1(y) + P_1(x) P_3(y)
# 2P_2(x)P_2(y)
########


struct PermutationInvariant{T,B} <: MultivariateOrthogonalPolynomial{2,T}
    basis::B
end
PermutationInvariant(B::AbstractQuasiMatrix{T}) where T = PermutationInvariant{T, typeof(B)}(B)

PermutationInvariant() = PermutationInvariant(Normalized(Legendre()))

axes(::PermutationInvariant) = (Inclusion(ChebyshevInterval() × ChebyshevInterval()), dihedralaxis(∞))




function getindex(Q::PermutationInvariant, 𝐱::SVector{2}, Kk::BlockIndex{1})
    x,y = 𝐱
    K,k = block(Kk), blockindex(Kk)
    ℓ = Int(K)-k
    μ = k-1
    (Q.basis[x,ℓ+1]Q.basis[y,μ+1]+Q.basis[x,μ+1]Q.basis[y,ℓ+1])
end

getindex(Q::PermutationInvariant, 𝐱::SVector{2}, k::Int) = Q[𝐱,findblockindex(axes(Q,2),k)]

getindex(Q::PermutationInvariant, 𝐱::SVector{2}, J::Block{1}) = [Q[𝐱,J[j]] for j = 1:length(axes(Q,2)[J])]
getindex(Q::PermutationInvariant, 𝐱::SVector{2}, JR::BlockOneTo) = mortar([Q[𝐱,J] for J in JR])


struct PermutationKronVector{T,D<:AbstractVector{T}} <: AbstractBlockVector{T}
    d::D
end

axes(::PermutationKronVector) = (dihedralaxis(∞),)
size(::PermutationKronVector) = (ℵ₀,)


function getindex(D::PermutationKronVector, K::Block{1})
    K̃ = Int(K)
    D.d[1:K̃] .* D.d[2K̃-1:-1:K̃]    
end
getindex(D::PermutationKronVector, Kk::BlockIndex{1}) = D[block(Kk)][blockindex(Kk)]
getindex(D::PermutationKronVector, k::Int) = D[findblockindex(axes(D,1), k)]


function grammatrix(Q::PermutationInvariant)
    M = grammatrix(Q.basis)
    Diagonal(PermutationKronVector(M.diag))
end

@simplify function *(Ac::QuasiAdjoint{<:Any,<:PermutationInvariant}, B::PermutationInvariant)
    M = (Ac').basis'B.basis
    Diagonal(PermutationKronVector(M.diag))
end
