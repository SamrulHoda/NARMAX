using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov,
                  collectAverageEnergyInbounds, localEdgeToRegion, ultimatePartner, region, Region, isClamped, currentInferenceAlgorithm
import SpecialFunctions: digamma
export NAutoRegressiveMovingAverageX, NARMAX

"""
Description:

    A Nonlinear AutoRegressive model with moving average and exogeneous input (NARMAX)

    y_k = fθ(u_k, …, u_k-M1, y_k-1, …, y_k-M2, , e_k, …, e_k-M3)

    where M1 is the number of input delays, M2 the number of output delays
    and M3 the number of error delays. These histories are stored as the following 
    vectors:
    - x_k-1 = [u_k-1, …, u_k-M1]
    - z_k-1 = [y_k-1, …, y_k-M2] 
    - r_k-1 = [e_k-1, …, e_k-M3]

    Assume u_k, x_k-1, z_k-1 and r_k-1 are observed and e_k ~ N(0, τ^-1).

    !! Currently, fθ is assumed to be a linear product of coefficients θ 
    and a basis expansion ϕ of inputs, outputs and errors: 
    
        fθ(...) = θ'*ϕ(u_k, …, u_k-M1, y_k-1, …, y_k-M2, , e_k, …, e_k-M3)

Interfaces:

    1. y (output)
    2. θ (function coefficients)
    3. u (current input)
    4. x (previous inputs)
    5. z (previous outputs)
    6. r (previous errors)
    7. τ (precision)

Construction:

    NAutoRegressiveMovingAverageX(y, θ, u, x, z, r, τ, g=ϕ, id=:some_id)
"""

mutable struct NAutoRegressiveMovingAverageX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    ϕ::Function # Scalar function between autoregression coefficients and state variable

    function NAutoRegressiveMovingAverageX(y, θ, u, x, z, r, τ; ϕ::Function, id=generateId(NAutoRegressiveMovingAverageX))
        @ensureVariables(y, θ, x, u, z, r, τ)
        self = new(id, Array{Interface}(undef, 7), Dict{Symbol,Interface}(), ϕ)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:θ] = self.interfaces[2] = associate!(Interface(self), θ)
        self.i[:u] = self.interfaces[3] = associate!(Interface(self), u)
        self.i[:x] = self.interfaces[4] = associate!(Interface(self), x)
        self.i[:z] = self.interfaces[5] = associate!(Interface(self), z)
        self.i[:r] = self.interfaces[6] = associate!(Interface(self), r)
        self.i[:τ] = self.interfaces[7] = associate!(Interface(self), τ)
        return self
    end
end

slug(::Type{NAutoRegressiveMovingAverageX}) = "NARMAX"

function averageEnergy(::Type{NAutoRegressiveMovingAverageX},
                       ϕ::Function,
                       marg_y::ProbabilityDistribution{Univariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_u::ProbabilityDistribution{Univariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_z::ProbabilityDistribution{Multivariate},
                       marg_r::ProbabilityDistribution{Multivariate},
                       marg_τ::ProbabilityDistribution{Univariate})

    # Extract moments of beliefs
    mθ,Vθ = unsafeMeanCov(marg_θ)
    my = unsafeMean(marg_y)
    mu = unsafeMean(marg_u)
    mx = unsafeMean(marg_x)
    mz = unsafeMean(marg_z)
    mr = unsafeMean(marg_r)
    mτ = unsafeMean(marg_τ)  
    aτ = marg_τ.params[:a]
    bτ = marg_τ.params[:b]

    # Basis expansion applied to moments
    ϕm = ϕ([mu; mx; mz; mr])

    # Return E_q -log p(y_k | u_k, .., θ, τ)
    return 1/2*log(2*pi) - 1/2*(digamma(aτ)-log(bτ)) + 1/2*mτ*(my^2 - 2*my*(mθ'*ϕm) + (mθ'*ϕm)^2 + ϕm'*Vθ*ϕm)
end

function collectAverageEnergyInbounds(node::NAutoRegressiveMovingAverageX)
    inbounds = Any[]

    # Push function to calling signature (ϕ needs to be defined in user scope)
    push!(inbounds, Dict{Symbol, Any}(:ϕ => node.ϕ, :keyword => false))

    local_edge_to_region = localEdgeToRegion(node)

    encountered_regions = Region[] # Keep track of encountered regions
    for node_interface in node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        current_region = region(node_interface.node, node_interface.edge)

        if isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(copy(inbound_interface.node), ProbabilityDistribution)) # Copy Clamp before assembly to prevent overwriting dist_or_msg field
        elseif !(current_region in encountered_regions)
            # Collect marginal entry from marginal dictionary (if marginal entry is not already accepted)
            target = local_edge_to_region[node_interface.edge]
            current_inference_algorithm = currentInferenceAlgorithm()
            push!(inbounds, current_inference_algorithm.target_to_marginal_entry[target])
        end

        push!(encountered_regions, current_region)
    end

    return inbounds
end
