####
# we try to understand dihedral-adapted OPs from their invariant OPs.
#
# Let's start with the scalar irreps. We have a simple basis of dihedral invariant OPs:

using SymmetricOrthogonalPolynomials, ClassicalOrthogonalPolynomials
P = Normalized(Legendre())
Q = DihedralInvariant(P)