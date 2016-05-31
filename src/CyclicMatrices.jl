# ------------------------------------------------------------------- #
# Copyright 2015-2016, Davide Lasagna, AFM, University of Southampton #
# ------------------------------------------------------------------- #
__precompile__()
module CyclicMatrices

# TODO
# ~ implement @checkbounds mechanisms for v0.5

import Base: full, 
             size, 
             A_ldiv_B!, 
             copy,
             getindex, 
             setindex!

import Base.LinAlg: Factorization, 
                    At_ldiv_B!, 
                    lufact, 
                    lufact!

export CyclicMatrix,
       upper, 
       body, 
       lower

immutable CyclicMatrix{ T,
                       MB<:AbstractMatrix,
                       MA<:AbstractMatrix,
                       MC<:AbstractMatrix} <: AbstractMatrix{T}
    B::MB # main band
    A::MA # upper right block
    C::MC # lower left block
    function CyclicMatrix(B::AbstractMatrix{T},
                          A::AbstractMatrix{T},
                          C::AbstractMatrix{T})
        size(A) == size(C) ||
            throw(ArgumentError("size of A and C must match"))
        new(B, A, C)
    end
end
CyclicMatrix{T}(B::AbstractMatrix{T},
                A::AbstractMatrix{T},
                C::AbstractMatrix{T}) =
    CyclicMatrix{T, typeof(B), typeof(A), typeof(C)}(B, A, C)

# field getters
@inline upper(M::CyclicMatrix) = M.A
@inline body(M::CyclicMatrix)  = M.B
@inline lower(M::CyclicMatrix) = M.C

# size, eltype, length
size(M::CyclicMatrix) = size(M.B)
length(M::CyclicMatrix) = prod(size(M.B))
eltype{T}(M::CyclicMatrix{T}) = T

# basic indexing 
@inline _isupper(M, i, j) = (m = size(M.A, 1); i<=m && j>size(M, 2)-m)
@inline _islower(M, i, j) = (m = size(M.A, 1); j<=m && i>size(M, 1)-m)

function getindex(M::CyclicMatrix, i::Integer, j::Integer)
    _isupper(M, i, j) && return upper(M)[i, j - (size(M, 1) - size(M.A, 1))]
    _islower(M, i, j) && return lower(M)[i - (size(M, 1) - size(M.A, 1)), j]
    body(M)[i, j]
end

function setindex!{T}(M::CyclicMatrix{T}, val, i::Integer, j::Integer)
    _isupper(M, i, j) && (return upper(M)[i, j - (size(M, 1) - size(M.A, 1))] = val)
    _islower(M, i, j) && (return lower(M)[i - (size(M, 1) - size(M.A, 1)), j] = val)
    body(M)[i, j] = val
end

# copy and to dense storage
copy(M::CyclicMatrix) = CyclicMatrix(copy(M.B), copy(M.A), copy(M.C))

function full(M::CyclicMatrix)
    m = size(M.A, 1)
    Af = zeros(eltype(M), size(M)...)
    Af[:] = M.B
    Af[1:m, (end-m+1):end] = M.A
    Af[(end-m+1):end, 1:m] = M.C
    Af
end


# ~~~ Linear algebra ~~~ 

# parameters
const _α = 1
const _γ = 1

immutable CyclicMatrixLU{T, MBᶠ<:Factorization, MA, MC} <: Factorization{T}
    Bᶠ::MBᶠ
    A::MA
    C::MC
    UV::Matrix{T}
end
size(Mᶠ::CyclicMatrixLU) = size(Mᶠ.Bᶠ)
size(Mᶠ::CyclicMatrixLU, i::Integer) = size(Mᶠ.Bᶠ, i)

function lufact!{T, MB, MA, MC}(M::CyclicMatrix{T, MB, MA, MC})
    # block size
    m = size(M.A, 1)

    # aliases
    B, A, C = body(M), upper(M), lower(M)

    # modify the main body before factorisation
    Δ = size(M, 1) - m
    for j = 1:m, i = 1:m
        B[i,     j] -= _γ/_α*C[i, j]
        B[i+Δ, j+Δ] -= _α/_γ*A[i, j]
    end

    # allocate space for U or V
    UV = zeros(T, size(B, 1), m)

    # compute factorisation, in place if possible
    Bᶠ = method_exists(lufact!, (typeof(B), )) ? lufact!(B) : lufact(B)

    # construct object and return
    CyclicMatrixLU{T, typeof(Bᶠ), MA, MC}(Bᶠ, A, C, UV)
end


A_ldiv_B!{T}(M::CyclicMatrix{T}, b::AbstractVector{T}) = A_ldiv_B!(lufact!(M), b)
At_ldiv_B!{T}(M::CyclicMatrix{T}, b::AbstractVector{T}) = At_ldiv_B!(lufact!(M), b)


function At_ldiv_B!{T<:Number}(Mᶠ::CyclicMatrixLU{T}, b::AbstractVector)
    # aliases
    Bᶠ, A, C, V = Mᶠ.Bᶠ, Mᶠ.A, Mᶠ.C, Mᶠ.UV

    # block size
    m = size(A, 1)

    # define V
    fill!(V, zero(T))     
    Δ = size(V, 1) - m 
    for j = 1:m, i = 1:m
        @inbounds V[i,   j] = _γ*C[j, i] # these are the transposed  
        @inbounds V[Δ+i, j] = _α*A[j, i] # these are the transposed 
    end

    # solve Bᶠᵀy⁰ = f using Bᶠ - f is aliased to y⁰
    y⁰ = At_ldiv_B!(Bᶠ, b)

    # solve BᶠᵀZ⁰ = V, in place - V is aliased to Z
    Z⁰ = At_ldiv_B!(Bᶠ, V)

    M⁰ = eye(m, m)                                                  # allocation
    for j = 1:m 
        @simd for i = 1:m
            @inbounds M⁰[i, j] += Z⁰[i, j]/_α + Z⁰[i+Δ, j]/_γ
        end
    end

    # construct r⁰ = Uᵀy⁰ = y¹/α + yᵐ/γ
    r⁰ = Vector{T}(m)                                               # allocation 
    @simd for i = 1:m
        @inbounds r⁰[i] = y⁰[i]/_α + y⁰[Δ+i]/_γ
    end
    
    # solve M⁰u⁰ = r⁰ = Uᵀy⁰ - r⁰ is aliased to u⁰
    u⁰, _, _ = LAPACK.gesv!(M⁰, r⁰)

    # x⁰ = - Z⁰*u⁰ + y⁰
    LAPACK.BLAS.gemv!('N', -1.0, Z⁰, u⁰, 1.0, y⁰)           
end

function A_ldiv_B!{T<:Number}(Mᶠ::CyclicMatrixLU{T}, b::AbstractVector)
    # aliases
    Bᶠ, A, C, U = Mᶠ.Bᶠ, Mᶠ.A, Mᶠ.C, Mᶠ.UV

    # block size
    m = size(A, 1)

    # define U 
    fill!(U, zero(T))     
    @simd for i = 1:m
        @inbounds U[i,             i] = 1/_α
        @inbounds U[end-i+1, end-i+1] = 1/_γ
    end

    # solve Bᶠy = f using Bᶠ - f is aliased to y
    y = A_ldiv_B!(Bᶠ, b)

    # solve BᶠZ = U, in place - U is aliased to Z
    Z = A_ldiv_B!(Bᶠ, U)

    # construct M
    CZ¹ = C*Z[1:m,           :]                                     # allocation 
    AZⁿ = A*Z[(end-m+1):end, :]                                     # allocation

    M = eye(m, m)                                                   # allocation
    for j = 1:m 
        @simd for i = 1:m
            @inbounds M[i, j] += _γ*CZ¹[i, j] + _α*AZⁿ[i, j]
        end
    end

    # construct Vᵀy = _γ*C*y¹ + _α*A*yᵐ
    Vᵀy = C*y[1:m]*_γ + A*y[end-m+1:end]*_α                         # allocation 
    
    # solve Mu = Vᵀy - Vᵀy is aliased to u
    u, _, _ = LAPACK.gesv!(M, Vᵀy)

    # x = y - Z*u
    LAPACK.BLAS.gemv!('N', -1.0, Z, u, 1.0, y)           
end

end