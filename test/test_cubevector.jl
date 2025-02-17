using LinearAlgebra, DynamicPolynomials, NumericalRepresentationTheory, LazyBandedMatrices
import SymmetricOrthogonalPolynomials: cubegen1, cubegen2, cubegen3


@polyvar(x,y,z)


Q = Array{Matrix{Float64}}(undef, 10)


ℙ₂ = (n,x,y) -> [legendrep(n-k,x)legendrep(k,y) for k=0:n]
ℙ = (n,x,y,z) -> vcat([legendrep(n-k,x) .* ℙ₂(k,y,z) for k=0:n]...)

𝕍 = (n,x,y,z) -> hcat(vcat.(ℙ(n,x,y,z), 0,0)..., vcat.(0, ℙ(n,x,y,z),0)...,vcat.(0, 0, ℙ(n,x,y,z))...)

𝕍(2,x,y,z)


s₁ = [0 0 1;
0 -1 0;
1 0 0]

s₂ = [-1 0 0;
0 0 -1;
0 -1 0]

s₃ = [0 0 -1;
0 -1 0;
-1 0 0]

N = 5
Qs = Array{Matrix{Float64}}(undef, N)
for n = 0:N-1
    m = size(𝕍(n,x,y,z),2)
    S₁ = BlockArray(zeros(Int,m,m), Fill(m÷3,3), Fill(m÷3,3))
    S₁[Block(1,3)] = cubegen1(n+1)
    S₁[Block(2,2)] = -cubegen1(n+1)
    S₁[Block(3,1)] = cubegen1(n+1)
    S₂ = BlockArray(zeros(Int,m,m), Fill(m÷3,3), Fill(m÷3,3))
    S₂[Block(1,1)] = -cubegen2(n+1)
    S₂[Block(2,3)] = -cubegen2(n+1)
    S₂[Block(3,2)] = -cubegen2(n+1)
    S₃ = BlockArray(zeros(Int,m,m), Fill(m÷3,3), Fill(m÷3,3))
    S₃[Block(3,1)] = -cubegen3(n+1)
    S₃[Block(2,2)] = -cubegen3(n+1)
    S₃[Block(1,3)] = -cubegen3(n+1)


    @test s₁*𝕍(n,s₁*[x,y,z]...) == 𝕍(n,x,y,z) * Matrix(S₁')
    @test s₂*𝕍(n,s₂*[x,y,z]...) == 𝕍(n,x,y,z) * Matrix(S₂')
    @test s₃*𝕍(n,s₃*[x,y,z]...) == 𝕍(n,x,y,z) * Matrix(S₃')


    @test S₁^2 == S₂^2 == S₃^2 == I
    @test S₁*S₃ == S₃*S₁
    @test (S₁*S₂)^3 == (S₂*S₃)^3 == I
    ρ,Qs[n+1] = blockdiagonalize(Representation(Matrix{Int}[S₁,S₂,S₃]))
end


Q = blockdiag(sparse.(Qs)...)
P = Legendre()
D¹ = (P'diff(P))[1:N,1:N]
D² = (diff(P)'diff(P))[1:N,1:N]
M = (P'P)[1:N,1:N]


D_xy = sparse(KronTrav(D¹, D¹', M))
D_xz = sparse(KronTrav(D¹, M, D¹'))
D_xy = sparse(KronTrav(D¹, D¹', M))
D_yz = sparse(KronTrav(M, D¹, D¹'))
D_xx = sparse(KronTrav(D², M, M))
D_yy = sparse(KronTrav(M, D², M))
D_zz = sparse(KronTrav(M, M, D²))

L = [D_yy+D_zz  -D_xy      -D_xz; 
     -D_xy'      D_xx+D_zz -D_yz; 
     -D_xz'     -D_yz'      D_xx+D_yy]

@test L ≈ L'