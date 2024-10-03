module SymmetricOrthogonalPolynomials
using BlockArrays

export dihedralQ, dihedral_signfilter, dihedral_trivialfilter, dihedral_tsfilter, dihedral_stfilter, dihedral_faithfulfilter1, dihedral_faithfulfilter2
export reflection_trivialfilter, reflection_signfilter, reflection_tsfilter, reflection_stfilter

include("dihedral.jl")
include("reflection.jl")

end # module
