"Utility functions"

using LinearAlgebra
import SpecialFunctions: digamma

function cholinv(M::AbstractMatrix)
    return LinearAlgebra.inv(M + 1e-8*Matrix{Float64}(I,size(M)))
    # return LinearAlgebra.inv(M)
end

function wMatrix(γ, order)
    mW = 1e8*Matrix{Float64}(I, order, order)
    mW[end, end] = γ
    return mW
end

function mvdigamma(x, order)
    "Multivariate digamma function (see https://en.wikipedia.org/wiki/Multivariate_gamma_function)"
    result = 0.
    for i = 1:order
        result += digamma(x + (1 - i)/2.)
    end
    return result
end

function polynomialExpansionMatrix(degree::Int, length_basis::Int)
    "Generate polynomial basis expansion of a certain degree."

    # Start combinations vector
    M = collect(0:degree)'
    for n = 2:length_basis

        # Generate combinations
        M = [repeat(M, outer=(1,degree+1)); kron(collect(0:degree)', ones(1,size(M,2)))]
        
        # remove combinations which have a too high degree
        mask = sum(M, dims=1)[:] .<= degree
        M = M[:, mask]
    end
    return M
end

function polynomialExpansion(x, degree::Int; input_delay::Int=0, output_delay::Int=0, error_delay::Int=0)
    "Generate polynomial basis expansion of a certain degree."

    # Total length of basis vector
    nk = input_delay + output_delay + error_delay + 1

    # Define basis expansion matrix
    M = polynomialExpansionMatrix(degree, nk)

    # Return expansion
    return [prod(x.^M[:,k]) for k in 1:size(M,2)]
end