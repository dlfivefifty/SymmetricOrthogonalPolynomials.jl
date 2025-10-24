using ClassicalOrthogonalPolynomials, SymmetricOrthogonalPolynomials, StaticArrays, BlockArrays, Test, BlockBandedMatrices

@testset "ChebyshevU" begin
    U = ChebyshevU()
    P = PermutationInvariant(U)

    x,y = 𝐱 = SVector(0.1,0.2)
    @testset "formula" begin
        @test P[𝐱,Block(1)[1]] == 2
        @test P[𝐱,Block(2)[1]] == U[x,2]U[y,1] + U[x,1]U[y,2]
        @test P[𝐱,Block(3)] == [U[x,3]U[y,1] + U[x,1]U[y,3], 2U[x,2]U[y,2]]
        @test P[𝐱,Block(4)] == [U[x,4]U[y,1] + U[x,1]U[y,4], U[x,3]U[y,2] + U[x,2]U[y,3]]
    end

    @testset "x+y Jacobi" begin
        @test x * U[x,1] == U[x,2]/2
        @test (x+y)*P[𝐱,Block(1)[1]] == P[𝐱,Block(2)[1]]
        @test (x+y)*P[𝐱,Block(2)[1]] ≈ U[x,1]U[y,1] + U[x,3]U[y,1]/2 + U[x,2]U[y,2] + U[x,1]*U[y,3]/2 ≈ (P[𝐱,Block(1)[1]] + P[𝐱,Block(3)[1]] + P[𝐱,Block(3)[2]])/2
        @test (x+y)*P[𝐱,Block(3)[1]] ≈ U[x,2]U[y,1]/2 + U[x,4]U[y,1]/2 + U[x,2]U[y,3]/2 + U[x,3]U[y,2]/2 + U[x,1]U[y,2]/2 + U[x,1]U[y,4]/2 ≈ (P[𝐱,Block(2)[1]] + P[𝐱,Block(4)[1]] + P[𝐱,Block(4)[2]])/2
        @test (x+y)*P[𝐱,Block(3)[2]] ≈ U[x,1]U[y,2] + U[x,3]U[y,2] + U[x,2]U[y,1] + U[x,2]U[y,3] ≈ P[𝐱,Block(2)[1]] + P[𝐱,Block(4)[2]]
        @test (x+y)*P[𝐱,Block(4)[1]] ≈ (P[𝐱,Block(3)[1]] + P[𝐱,Block(5)[1]] + P[𝐱,Block(5)[2]])/2
        @test (x+y)*P[𝐱,Block(4)[2]] ≈ U[x,2]U[y,2]/2 + U[x,4]U[y,2]/2 + U[x,1]U[y,3]/2 + U[x,3]U[y,3]/2 + y * (U[x,3]U[y,2] + U[x,2]U[y,3]) ≈  (P[𝐱,Block(3)[1]]+P[𝐱,Block(3)[2]] + P[𝐱,Block(5)[2]]+P[𝐱,Block(5)[3]])/2
        @test (x+y)*P[𝐱,Block(5)[1]]  ≈  (P[𝐱,Block(4)[1]]+ P[𝐱,Block(6)[1]]+P[𝐱,Block(6)[2]])/2
        @test (x+y)*P[𝐱,Block(5)[2]] ≈ (P[𝐱,Block(4)[1]]+P[𝐱,Block(4)[2]] + P[𝐱,Block(6)[2]]+P[𝐱,Block(6)[3]])/2
        @test (x+y)*P[𝐱,Block(5)[3]] ≈ P[𝐱,Block(4)[2]] + P[𝐱,Block(6)[4]]
        @test (x+y)*P[𝐱,Block(6)[1]]  ≈  (P[𝐱,Block(5)[1]]+ P[𝐱,Block(7)[1]]+P[𝐱,Block(7)[2]])/2
        @test (x+y)*P[𝐱,Block(6)[2]] ≈ (P[𝐱,Block(5)[1]]+P[𝐱,Block(5)[2]] + P[𝐱,Block(7)[2]]+P[𝐱,Block(7)[3]])/2
        @test (x+y)*P[𝐱,Block(6)[3]] ≈ (P[𝐱,Block(5)[2]]+P[𝐱,Block(5)[3]] + P[𝐱,Block(7)[3]]+P[𝐱,Block(7)[4]])/2


        J = (BandedBlockBandedMatrix{Float64}(undef, (axes(P,2)[Block.(Base.OneTo(7))], axes(P,2)[Block.(Base.OneTo(7))]), (1,1), (1,1)) .= 0)

        J.data[1:2,:] .= 1/2
        J.data[8:9,:] .= 1/2

        for K = 1:2:7
            J.data[Block(1,K)[1,K÷2+1]] = 1
            J.data[Block(3,K)[2,K÷2+1]] = 1
        end

        @test (x+y)*P[𝐱,Block.(1:6)]' ≈ P[𝐱,Block.(1:7)]'*J[:,Block.(1:6)]

        ### renormalize
        m = J[:,1]
        m .= 1
        m[1] = sqrt(2)
        # m[Block(3)[2]] =
        Diagonal(m)*J*Diagonal(inv.(m))
    end

    @testset "xy Jacobi" begin
        @test (x*y)*P[𝐱,Block(1)[1]] == P[𝐱,Block(3)[2]]/4
        @test (x*y)*P[𝐱,Block(2)[1]] ≈ U[x,1]U[y,2]/4 + U[x,3]U[y,2]/4 + U[x,2]U[y,1]/4 + U[x,2]U[y,3]/4 ≈
                                        P[𝐱,Block(2)[1]]/4 + P[𝐱,Block(4)[2]]/4
        @test (x*y)*P[𝐱,Block(3)[1]] ≈ U[x,2]U[y,2]/2 + U[x,4]U[y,2]/4 + U[x,2]U[y,4]/4 ≈
                                        P[𝐱,Block(3)[2]]/4 + P[𝐱,Block(5)[2]]/4
        @test (x*y)*P[𝐱,Block(3)[2]] ≈ U[x,1]U[y,1]/2 + U[x,1]U[y,3]/2 + U[x,3]U[y,1]/2 + U[x,3]U[y,3]/2 ≈
                                        P[𝐱,Block(1)[1]]/4 + P[𝐱,Block(3)[1]]/2+ P[𝐱,Block(5)[3]]/4
        @test (x*y)*P[𝐱,Block(4)[1]] ≈ P[𝐱,Block(4)[2]]/4 + P[𝐱,Block(6)[2]]/4
        @test (x*y)*P[𝐱,Block(4)[2]] ≈ U[x,2]U[y,1]/4 + U[x,1]U[y,2]/4 + U[x,2]U[y,3]/4 + U[x,3]U[y,2]/4 + U[x,4]U[y,1]/4 + U[x,1]U[y,4]/4 + U[x,4]U[y,3]/4 + U[x,3]U[y,4]/4 ≈
                                        P[𝐱,Block(2)[1]]/4 + P[𝐱,Block(4)[2]]/4 + P[𝐱,Block(4)[1]]/4 + P[𝐱,Block(6)[3]]/4
        @test (x*y)*P[𝐱,Block(5)[1]] ≈ P[𝐱,Block(5)[2]]/4 + P[𝐱,Block(7)[2]]/4
        @test (x*y)*P[𝐱,Block(5)[2]] ≈ U[x,3]U[y,1]/4 + U[x,1]U[y,3]/4 + U[x,3]U[y,3]/2 + U[x,5]U[y,1]/4 + U[x,1]U[y,5]/4 + U[x,5]U[y,3]/4 + U[x,3]U[y,5]/4 ≈
                                        P[𝐱,Block(3)[1]]/4 + P[𝐱,Block(5)[1]]/4 + P[𝐱,Block(5)[3]]/4 + P[𝐱,Block(7)[3]]/4
        @test (x*y)*P[𝐱,Block(5)[3]] ≈ P[𝐱,Block(3)[2]]/4 + P[𝐱,Block(5)[2]]/2+ P[𝐱,Block(7)[4]]/4
        @test (x*y)*P[𝐱,Block(6)[1]] ≈ P[𝐱,Block(6)[2]]/4 + P[𝐱,Block(8)[2]]/4
        @test (x*y)*P[𝐱,Block(6)[2]] ≈ U[x,4]U[y,1]/4 + U[x,4]U[y,3]/4 + U[x,6]U[y,1]/4 + U[x,6]U[y,3]/4 + x*y*U[x,2]U[y,5] ≈
                                        P[𝐱,Block(4)[1]]/4 + P[𝐱,Block(6)[1]]/4 + P[𝐱,Block(6)[3]]/4  + P[𝐱,Block(8)[3]]/4

        B = (BandedBlockBandedMatrix{Float64}(undef, (axes(P,2)[Block.(Base.OneTo(8))], axes(P,2)[Block.(Base.OneTo(8))]), (2,2), (1,1)) .= 0)
        B[Block(3,1)[2,1]] = 1/4
        B[Block(2,2)[1,1]]  = B[Block(4,2)[2,1]]  = 1/4;
        B[Block(3,3)[2,1]]  = B[Block(5,3)[2,1]]  = 1/4;
        B[Block(1,3)[1,2]] = B[Block(5,3)[3,2]]  = 1/4; B[Block(3,3)[1,2]]  = 1/2;
        B[Block(4,4)[2,1]] = B[Block(6,4)[2,1]]  = 1/4; 
        B[Block(2,4)[1,2]] = B[Block(4,4)[1,2]] = B[Block(4,4)[2,2]] = B[Block(6,4)[3,2]] = 1/4
        B[Block(5,5)[2,1]] = B[Block(7,5)[2,1]] = 1/4
        B[Block(3,5)[1,2]] = B[Block(5,5)[1,2]] = B[Block(5,5)[3,2]] = B[Block(7,5)[3,2]] = 1/4
        B[Block(3,5)[2,3]] = B[Block(7,5)[4,3]] = 1/4; B[Block(5,5)[2,3]] = 1/2
        B[Block(6,6)[2,1]] = B[Block(8,6)[2,1]]  = 1/4; 
        B[Block(4,6)[1,2]] = B[Block(6,6)[1,2]] = B[Block(6,6)[3,2]] = B[Block(8,6)[3,2]] = 1/4
        B[Block(4,6)[2,3]] = B[Block(6,6)[2,3]] = B[Block(6,6)[3,3]] = B[Block(8,6)[4,3]] = 1/4
        @test (x*y)*P[𝐱,Block.(1:6)]' ≈ P[𝐱,Block.(1:7)]'*B[:,Block.(1:6)]
        (x*y)*P[𝐱,Block.(1:6)]'
        P[𝐱,Block.(1:8)]'*B[:,Block.(1:6)]
    end
end

@testset "ChebyshevT" begin
    T = ChebyshevT()
    P = PermutationInvariant(T)

    x,y = 𝐱 = SVector(0.1,0.2)
    @testset "formula" begin
        @test P[𝐱,Block(1)[1]] == 2
        @test P[𝐱,Block(2)[1]] == T[x,2]T[y,1] + T[x,1]T[y,2]
        @test P[𝐱,Block(3)] == [T[x,3]T[y,1] + T[x,1]T[y,3], 2T[x,2]T[y,2]]
        @test P[𝐱,Block(4)] == [T[x,4]T[y,1] + T[x,1]T[y,4], T[x,3]T[y,2] + T[x,2]T[y,3]]
    end

    @testset "first Jacobi" begin
        @test x * T[x,1] == T[x,2]
        @test (x+y)*P[𝐱,Block(1)[1]] == 2P[𝐱,Block(2)[1]]
        @test (x+y)*P[𝐱,Block(2)[1]] ≈ T[x,1]T[y,1] + T[x,3]T[y,1]/2 + 2T[x,2]T[y,2] + T[x,1]*T[y,3]/2 ≈ (P[𝐱,Block(1)[1]] + P[𝐱,Block(3)[1]])/2 + P[𝐱,Block(3)[2]]
        @test (x+y)*P[𝐱,Block(3)[1]] ≈ T[x,2]T[y,1]/2 + T[x,4]T[y,1]/2 + T[x,2]T[y,3] + T[x,3]T[y,2] + T[x,1]T[y,2]/2 + T[x,1]T[y,4]/2 ≈ (P[𝐱,Block(2)[1]] + P[𝐱,Block(4)[1]])/2 + P[𝐱,Block(4)[2]]
        @test (x+y)*P[𝐱,Block(3)[2]] ≈ T[x,1]T[y,2] + T[x,3]T[y,2] + T[x,2]T[y,1] + T[x,2]T[y,3] ≈ P[𝐱,Block(2)[1]] + P[𝐱,Block(4)[2]]
        @test (x+y)*P[𝐱,Block(4)[1]] ≈ (P[𝐱,Block(3)[1]] + P[𝐱,Block(5)[1]])/2 + P[𝐱,Block(5)[2]]
        @test (x+y)*P[𝐱,Block(4)[2]] ≈  (P[𝐱,Block(3)[1]]+P[𝐱,Block(3)[2]] + P[𝐱,Block(5)[2]])/2 + P[𝐱,Block(5)[3]]
        @test (x+y)*P[𝐱,Block(5)[1]]  ≈  (P[𝐱,Block(4)[1]]+ P[𝐱,Block(6)[1]]+P[𝐱,Block(6)[2]])/2
        @test (x+y)*P[𝐱,Block(5)[2]] ≈ (P[𝐱,Block(4)[1]]+P[𝐱,Block(4)[2]] + P[𝐱,Block(6)[2]]+P[𝐱,Block(6)[3]])/2
        @test (x+y)*P[𝐱,Block(5)[3]] ≈ P[𝐱,Block(4)[2]] + P[𝐱,Block(6)[4]]
        @test (x+y)*P[𝐱,Block(6)[1]]  ≈  (P[𝐱,Block(5)[1]]+ P[𝐱,Block(7)[1]]+P[𝐱,Block(7)[2]])/2
        @test (x+y)*P[𝐱,Block(6)[2]] ≈ (P[𝐱,Block(5)[1]]+P[𝐱,Block(5)[2]] + P[𝐱,Block(7)[2]]+P[𝐱,Block(7)[3]])/2
        @test (x+y)*P[𝐱,Block(6)[3]] ≈ (P[𝐱,Block(5)[2]]+P[𝐱,Block(5)[3]] + P[𝐱,Block(7)[3]]+P[𝐱,Block(7)[4]])/2
    end

    J = (BandedBlockBandedMatrix{Float64}(undef, (axes(P,2)[Block.(Base.OneTo(7))], axes(P,2)[Block.(Base.OneTo(7))]), (1,1), (1,1)) .= 0)

    J.data[1:2,:] .= 1/2
    J.data[8:9,:] .= 1/2

    for K = 1:2:7
        J.data[Block(1,K)[1,K÷2+1]] = 1
        J.data[Block(3,K)[2,K÷2+1]] = 1
    end
    J[2,1] = 2
    for K = 2:6
        J[Block(K+1,K)[2,1]] = 1
    end


    @test (x+y)*P[𝐱,Block.(1:6)]' ≈ P[𝐱,Block.(1:7)]'*J[:,Block.(1:6)]

    ### renormalize
    m = J[:,1]
    m .= 1
    m[1] = 2
    m[Block(4)[2]] = 1/sqrt(2)
    Diagonal(m)*J*Diagonal(inv.(m))
end