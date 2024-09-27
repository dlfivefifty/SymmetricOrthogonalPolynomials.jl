using ClassicalOrthogonalPolynomials, LazyBandedMatrices, LinearAlgebra, BlockArrays, NumericalRepresentationTheory, Test


@testset "Generators" begin
    # The symmetry group of the square is apparently S₄ × ℤ₂. It's irreps
    # should be just the tensor product of the irreps of S₄ and ℤ₂. Since
    # ℤ₂ has scalar irreps generated by 1 or -1 we should have generators of S₄
    # alongside their negation. Let's check this out.

    # First we need to figure out the generators, starting with rotation invariance.
    # The isometry between the cube group and S₄ comes from permutation of diagonals.
    # Label the 4 diagonals by (the span of)

    a = [1,1,1]; b = [1,-1,1]; c = [1,1,-1]; d = [1,-1,-1]

    # I.e. s₁ corresponds to permuting a and b, leaving c and d fixed. By inspection
    # we find the 3 generators:

    s₁ = [0 0 1;
        0 -1 0;
        1 0 0]

    s₂ = [-1 0 0;
        0 0 -1;
        0 -1 0]

    s₃ = [0 0 -1;
        0 -1 0;
        -1 0 0]

    @test det(s₁) == det(s₂) == det(s₃) == 1
    @test s₁ * [a b c d] == [b a -c -d] # swapping sign is fine
    @test s₂ * [a b c d] == [-a -c -b -d]
    @test s₃ * [a b c d] == [-a -b d c]

    # We can also check the generator relationships directly

    @test s₁^2 == s₂^2 == s₃^2 == I
    @test s₁*s₃ == s₃*s₁
    @test (s₁*s₂)^3 == (s₂*s₃)^3 == I

    # OPs generate representations for these group actions


    P = legendrep
    Q = Array{Matrix{Float64}}(undef, 10)


    ℙ₂ = (n,x,y) -> [P(n-k,x)P(k,y) for k=0:n]
    ℙ = (n,x,y,z) -> vcat([P(n-k,x) .* ℙ₂(k,y,z) for k=0:n]...)
    ℚ = (n, x, y, z) -> Q[n]'*ℙ(n, x, y, z)

    x,y,z = 0.1,0.2,0.3
    n = 1; 
    @test ℙ(n,  z, -y,  x) ≈ s₁ * ℙ(n, x, y, z)
    @test ℙ(n, -x, -z, -y) ≈ s₂ * ℙ(n, x, y, z)
    @test ℙ(n, -z, -y, -x) ≈ s₃ * ℙ(n, x, y, z)
    Q[1] = I(3)

    # this is already an irrep
    @test multiplicities(Representation([s₁,s₂,s₃]))[Partition([2,1,1])] == 1


    n = 2
    S₁ = sparse([6,5,3,4,2,1], 1:6, [1,-1,1,1,-1,1])
    S₂ = sparse([1,3,2,6,5,4], 1:6, [1,1,1,1,1,1]) 
    S₃ = sparse([6,5,3,4,2,1], 1:6, [1,1,1,1,1,1])
    @test ℙ(n,  z, -y,  x) ≈ S₁ * ℙ(n,  x, y, z)
    @test ℙ(n, -x, -z, -y) ≈ S₂ * ℙ(n, x, y, z)
    @test ℙ(n, -z, -y, -x) ≈ S₃ * ℙ(n, x, y, z)

    ρ,Q[n] = blockdiagonalize(Representation([S₁,S₂,S₃]))
    @test Q[n]'Q[n] ≈ I

    σ₁,σ₂,σ₃ = ρ.generators
    @test ℚ(n,  z, -y,  x) ≈ σ₁*ℚ(n,  x, y, z)
    @test ℚ(n, -x, -z, -y) ≈  σ₂*ℚ(n, x, y, z)
    @test ℚ(n, -z, -y, -x) ≈  σ₃*ℚ(n, x, y, z)

    n = 3
    S₁ = sparse([10,9,6,8,5,3,7,4,2,1], 1:10, [1,-1,1,1,-1,1,-1,1,-1,1])
    S₂ = sparse([1,3,2,6,5,4,10,9,8,7], 1:10, -ones(Int,10)) 
    S₃ = sparse([10,9,6,8,5,3,7,4,2,1], 1:10, -ones(Int,10))
    @test ℙ(n,  z, -y,  x) ≈ S₁ * ℙ(n,  x, y, z)
    @test ℙ(n, -x, -z, -y) ≈ S₂ * ℙ(n, x, y, z)
    @test ℙ(n, -z, -y, -x) ≈ S₃ * ℙ(n, x, y, z)

    ρ,Q[n] = blockdiagonalize(Representation([S₁,S₂,S₃]))
    @test Q[n]'Q[n] ≈ I

    σ₁,σ₂,σ₃ = ρ.generators
    @test ℚ(n,  z, -y,  x) ≈ σ₁*ℚ(n,  x, y, z)
    @test ℚ(n, -x, -z, -y) ≈  σ₂*ℚ(n, x, y, z)
    @test ℚ(n, -z, -y, -x) ≈  σ₃*ℚ(n, x, y, z)



    # we now add in the ℤ₂ symmetry which should be (x,y,z) -> (-x,-y,-z)
    # note this is a reflection + rotation but clearly we can get the basic
    # reflections. E.g. the reflection (x,y,-z) can be produced as
    #

    @test -(s₁*s₂*s₃)^2 ≈ Diagonal([1,1,-1])

    #
    # The key point is it commutes with the rotations in the same way
    # group products commute.
    # We do a two stage process, first reduce to irreps of S₄. If we have
    # only one copy of an irrep then we have that it is also an irrep of ℤ₂


    n = 1; 
    ℙ(n,  -x, -y, -z) ≈ -ℙ(n,  -x, -y, -z) # sign rep


    n = 2
    @test ℚ(n, -x, -y, -z) ≈ ℚ(n, x, y, z)

    n = 3
    @test ℚ(n, -x, -y, -z) ≈ -ℚ(n, x, y, z)
end

@testset "Expansion" begin
    P = Legendre()
    N = 30
    Pl = plan_transform(P, (N,N,N))
    𝐱,𝐲,𝐳 = ClassicalOrthogonalPolynomials.grid(P, (N,N,N))
    f = (x,y,z) -> exp(-x^2 -2y^2 - 3(z-0.1)^2)
    F = f.(𝐱,𝐲', reshape(𝐳,1,1,:))
    C = Pl*F
    
    x,y,z = 0.1,0.2,0.3
    @test sum(P[y,1:N]' .* C .* P[x,1:N] .* reshape(P[z,1:N],1,1,n)) ≈ f(x,y,z)
    @test KronTrav(P[z,1:N], P[y,1:N], P[x,1:N])' * DiagTrav(C) ≈ f(x,y,z)


    function genrep(sym, n)
        ret = zeros(sum(1:n),sum(1:n))
        ℓ = 1
        for k = 1:n, j=1:k
            𝐏 = (x,y,z) -> P[x,n-k+1] * P[y,k-j+1] * P[z,j]
            𝐏̃ = (x,y,z) -> 𝐏(sym(x,y,z)...)
        ret[:,ℓ] = DiagTrav(Pl* 𝐏̃.(𝐱,𝐲',reshape(𝐳,1,1,:)))[Block(n)]
        ℓ += 1
        end
        ret
    end

    function blockdiagonalizepoly(n)
        τ1,τ2,τ3 = genrep((x,y,z) -> (z,-y,x), n), genrep((x,y,z) -> (-x, -z, -y), n), genrep((x,y,z) -> (-z, -y, -x), n)
        blockdiagonalize(Representation([τ1,τ2,τ3]))[2]
    end

    n = 3
    Q = blockdiagonalizepoly(n)
    s =  genrep((x,y,z) -> (-x,-y,-z), n)
    @test Q's*Q ≈ I
    
    n = 4
    Q = blockdiagonalizepoly(n)
    s =  genrep((x,y,z) -> (-x,-y,-z), n)
    @test Q's*Q ≈ -I
end

@testset "Poisson" begin
    N = 10
    Δ = (diff(P)'diff(P))[1:N,1:N]
    M = (P'P)[1:N,1:N]

    L = sparse(KronTrav(Δ, M, M)) + sparse(KronTrav(M, Δ, M)) + sparse(KronTrav(M, M, Δ))

    Qs = [blockdiagonalizepoly(n) for n=1:N]
    Q = blockdiag(sparse.(Qs)...)

    sparse(round.(Matrix(Q'*L*Q);digits=7))

    A = round.(Matrix(Q'*L*Q);digits=7)


    sparse9[:,end]
end


@testset "Schrodinger" begin
    x = axes(P,1)
    N = 10
    Δ = (diff(P)'diff(P))[1:N,1:N]
    M = (P'P)[1:N,1:N]
    X² = ((P'P) * (P\(x.^2 .* P)))[1:N,1:N]


    V = sparse(KronTrav(X², M, M)) + sparse(KronTrav(M, X², M)) + sparse(KronTrav(M, M, X²))

    A = L + V
    spy(round.(A;digits=7))
    spy(round.(Matrix(Q'*A*Q);digits=7))
end