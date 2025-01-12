using MultivariateOrthogonalPolynomials, InfiniteArrays, LazyArrays, BlockArrays
using ContinuumArrays: Basis

########
# Invariant polynomials with respect to D_4 are given by
# P_{2k}(x) P_{2j}(y) +  P_{2j}(x) P_{2k}(y)
# The first few are 
# 1
# -----
# P_2(x) + P_2(y)
# ----
# P_4(x) + P_4(y)
# P_2(x) P_2(y)
# -----
# P_6(x) + P_6(y)
# P_4(x)P_2(y) + P_2(x) P_4(y)
# ----
# P_8(x) + P_8(y)
# P_6(x)P_2(y) + P_2(x)P_6(y)
# P_4(x)P_4(y)
########


struct DihedralInvariantPolynomial{T} <: Basis{T} end

BlockedOneTo((2:∞) .^ 2 .÷ 4)