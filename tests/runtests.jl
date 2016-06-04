using Base.Test
using CyclicMatrices


let
    srand(123)
    for rep = 1:5
        N = rand(2:16)
        M = rand(50:100)
        vecs = Vector{Float64}[]
        poss = Int[]
        for i = -3N:3N
            L = (N*M) - abs(i)
            push!(vecs, randn(L))
            push!(poss, i)
        end

        # cyclic matrix object
        B  = spdiagm(vecs, poss)
        A = sprand(N, N, 0.2)
        C = sprand(N, N, 0.2)
        A = CyclicMatrix(B, A, C)
        b = randn(size(A, 1))

        # dense storage
        Af = full(A)
        bf = copy(b)

        # a copy
        Ac = copy(A)
        bc = copy(b)

        # solve systems
        x     = A_ldiv_B!(A, b)
        xfact = A_ldiv_B!(Ac, bc)
        xexac = Af\bf

        # check
        @test xexac ≈ x
        @test xfact ≈ x
    end
end

const B = Float64[ 1  3   8  0  -1  0   0  0   0  0   0  0;
                   2  4   0  8   0 -1   0  0   0  0   0  0;
                  -8  0   4  2   8  0  -1  0   0  0   0  0;
                   0 -8   3  1   0  8   0 -1   0  0   0  0;
                   1  0  -8  0   1  2   8  0  -1  0   0  0;
                   0  1   0 -8   3  4   0  8   0 -1   0  0;
                   0  0   1  0  -8  0   1  4   8  0  -1  0;
                   0  0   0  1   0 -8   3  2   0  8   0  1;
                   0  0   0  0   1  0  -8  0   2  3   8  0;
                   0  0   0  0   0  1   0 -8   1  4   0  8;
                   0  0   0  0   0  0   1  0  -8  0   3  2;
                   0  0   0  0   0  0   0  1   0 -8   4  1]

const A = Float64[ 1  0 -8  0;
                   0  1  0 -8;
                   0  0  1  0;
                   0  0  0  1]

const C = Float64[-1  0  0  0;
                   0 -1  0  0;
                   8  0 -1  0;
                   0  8  0 -1]

const MF = Float64[ 1  3   8  0  -1  0   0  0   1  0  -8  0;
                    2  4   0  8   0 -1   0  0   0  1   0 -8;
                   -8  0   4  2   8  0  -1  0   0  0   1  0;
                    0 -8   3  1   0  8   0 -1   0  0   0  1;
                    1  0  -8  0   1  2   8  0  -1  0   0  0;
                    0  1   0 -8   3  4   0  8   0 -1   0  0;
                    0  0   1  0  -8  0   1  4   8  0  -1  0;
                    0  0   0  1   0 -8   3  2   0  8   0  1;
                   -1  0   0  0   1  0  -8  0   2  3   8  0;
                    0 -1   0  0   0  1   0 -8   1  4   0  8;
                    8  0  -1  0   0  0   1  0  -8  0   3  2;
                    0  8   0 -1   0  0   0  1   0 -8   4  1]

const ENORM = 1e-10

# match a particular case I need
let
    M = CyclicMatrix(copy(B), copy(A), copy(C))

    @test eltype(M)  == Float64
    @test size(M)    == (12, 12)
    @test size(M, 1) == 12
    @test size(M, 2) == 12

    # test full
    @test full(M) == MF

    # test setindex
    for i = 1:12, j = 1:12
        M[i, j] = i*j
    end

    # test getindex
    @test M == [i*j for i = 1:12, j=1:12]
end

# linear system
let 
    M = CyclicMatrix(copy(B), copy(A), copy(C))
    xₑ = MF\collect(1.0:12.0)
    x  = A_ldiv_B!(lufact!(M), collect(1.0:12.0))
    @test norm(xₑ - x, Inf) < ENORM
end

# test transposed algorithm 1
let 
    M = CyclicMatrix(copy(B), copy(A), copy(C))
    xₑ = MF'\collect(1.0:12.0)
    x  = Base.LinAlg.At_ldiv_B!(lufact!(M), collect(1.0:12.0))
    @test norm(xₑ - x, Inf) < ENORM
end

# test transposed algorithm 2
let 
    M  = CyclicMatrix(copy(B), copy(A), copy(C))
    MT = CyclicMatrix(copy(B)', copy(C)', copy(A)')

    @test norm(Base.LinAlg.A_ldiv_B!(lufact!(M), collect(1.0:12.0)) -
               Base.LinAlg.At_ldiv_B!(lufact!(MT), collect(1.0:12.0)), Inf) < ENORM

    M  = CyclicMatrix(copy(B), copy(A), copy(C))
    MT = CyclicMatrix(copy(B)', copy(C)', copy(A)')

    @test norm(Base.LinAlg.A_ldiv_B!(lufact!(MT), collect(1.0:12.0)) -
               Base.LinAlg.At_ldiv_B!(lufact!(M), collect(1.0:12.0)), Inf) < ENORM
end

# test lufact makes a copy
let
   M = CyclicMatrix(copy(B), copy(A), copy(C))
   luM = lufact(M) 
   @test body(M)  == B
   @test upper(M) == A
   @test lower(M) == C
end