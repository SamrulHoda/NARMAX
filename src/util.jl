"Utility functions"

import ForneyLab: cholinv
import SpecialFunctions: digamma

function wMatrix(γ, order)
    mW = 1e8*Matrix{Float64}(I, order, order)
    mW[end, end] = γ
    return mW
end

function cholinv(M::AbstractMatrix)
    return LinearAlgebra.inv(M)
end

function mvdigamma(x, order)
    "Multivariate digamma function (see https://en.wikipedia.org/wiki/Multivariate_gamma_function)"
    result = 0.
    for i = 1:order
        result += digamma(x + (1 - i)/2.)
    end
    return result
end

function polynomialExpansion(x, degree; input_delay=0, output_delay=0, error_delay=0)
    "Generate polynomial basis expansion of a certain degree."

    # Total length of basis vector
    nk = input_delay + output_delay + error_delay + 1

    # Start combinations vector
    comb = collect(0:degree)';
    for n = 2:nk

        # Generate combinations
        comb = [repeat(comb, outer=(1,nd+1)); kron(collect(0:nd)', ones(1,size(comb,2)))]
        
        # remove combinations which have a too high degree
        mask = sum(comb, dims=1)[:] .<= nd
        comb = comb[:, mask]
    end

    return [prod(x.^comb[:,k]) for k in 1:size(comb,2)]
end