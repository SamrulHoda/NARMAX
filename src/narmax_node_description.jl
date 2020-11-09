using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov
export NAutoregressiveMovingAverageX, NARMAX

"""
Description:

    A Nonlinear Autoregressive model with moving average and exogeneous input (NARMAX)

    y_t = f(y_t-1, …, y_t-M, u_t, …, u_t-N, e_t, …, e_t-L)

    where M is the number of previous observations, N the number of previous inputs 
    and L the number of previous process noise estimates. 
    These histories are stored as the following vectors:
    - x_t-1 = [y_t-1, …, y_t-M]' 
    - z_t-1 = [u_t-1, …, u_t-N]'
    - γ_t-1 = [e_t-1, …, e_t-L]'.

    Assume y_t, x_t-1, z_t-1 and u_t are observed and e_t ~ N(0, τ^-1).

Interfaces:

    1. y (output)
    2. θ (function coefficients)
    3. x (previous observations vector)
    4. z (previous inputs vector)
    5. u (input)
    6. τ (precision)

Construction:

    NAutoregressiveMovingAverageX(y, θ, x, z, u, τ, g=f, id=:some_id)
"""

mutable struct NAutoregressiveMovingAverageX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    g::Function # Scalar function between autoregression coefficients and state variable

    function NAutoregressiveMovingAverageX(y, θ, x, u, z, τ, γ; g::Function, id=generateId(NAutoregressiveMovingAverageX))
        @ensureVariables(y, θ, x, u, z, τ, γ)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}(), g)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:θ] = self.interfaces[2] = associate!(Interface(self), θ)
        self.i[:x] = self.interfaces[3] = associate!(Interface(self), x)
        self.i[:u] = self.interfaces[4] = associate!(Interface(self), u)
        self.i[:z] = self.interfaces[5] = associate!(Interface(self), z)
        self.i[:τ] = self.interfaces[6] = associate!(Interface(self), τ)
        self.i[:γ] = self.interfaces[7] = associate!(Interface(self), γ)
        return self
    end
end

slug(::Type{NAutoregressiveMovingAverageX}) = "NARMAX"

function averageEnergy(::Type{NAutoregressiveMovingAverageX},
                       marg_y::ProbabilityDistribution{Univariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_u::ProbabilityDistribution{Univariate},
                       marg_z::ProbabilityDistribution{Multivariate},
                       marg_τ::ProbabilityDistribution{Univariate},
                       marg_γ::ProbabilityDistribution{Multivariate})

    error("not implemented yet")

end
