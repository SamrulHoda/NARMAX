import LinearAlgebra: I, inv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType,
				  collectNaiveVariationalNodeInbounds, assembleClamp!, ultimatePartner

include("util.jl")

export ruleVariationalNARMAXOutNPPPPPP,
       ruleVariationalNARMAXIn1PNPPPPP,
       ruleVariationalNARMAXIn2PPNPPPP,
       ruleVariationalNARMAXIn3PPPNPPP,
	   ruleVariationalNARMAXIn4PPPPNPP,
	   ruleVariationalNARMAXIn5PPPPPNP,
	   ruleVariationalNARMAXIn6PPPPPPN


function ruleVariationalNARMAXOutNPPPPPP(ϕ :: Function,
									     marg_y :: Nothing,
									     marg_θ :: ProbabilityDistribution{Multivariate},
                                         marg_u :: ProbabilityDistribution{Univariate},
										 marg_x :: ProbabilityDistribution{Multivariate},
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	mθ = unsafeMean(marg_θ)
	mu = unsafeMean(marg_u)
	mx = unsafeMean(marg_x)
	mz = unsafeMean(marg_z)
	mr = unsafeMean(marg_r)
	mτ = unsafeMean(marg_τ)

	# Basis function applied to moments
	ϕm = ϕ([mu; mx; mz; mr])

	# Set outgoing message
	return Message(Univariate, GaussianMeanPrecision, m=mθ'*ϕm, w=mτ)
end

function ruleVariationalNARMAXIn1PNPPPPP(ϕ :: Function,
									     marg_y :: ProbabilityDistribution{Univariate},
                                         marg_θ :: Nothing,
										 marg_u :: ProbabilityDistribution{Univariate},
									     marg_x :: ProbabilityDistribution{Multivariate},
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})

    # Extract moments of beliefs
	my = unsafeMean(marg_y)
	mu = unsafeMean(marg_u)
	mx = unsafeMean(marg_x)
	mz = unsafeMean(marg_z)
	mr = unsafeMean(marg_r)
	mτ = unsafeMean(marg_τ)

	# Basis function applied to moments
	ϕm = ϕ([mu; mx; mz; mr])

	# Number of coefficients
	num_coeffs = size(ϕm,1)

	# Update parameters
	# Ψ = mτ*ϕm*ϕm'
	Ψ = mτ*(ϕm*ϕm' .+ 1e-8*Matrix{Float64}(I, num_coeffs, num_coeffs))
	ψ = mτ*my*ϕm

    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ψ, w=Ψ)
end

function ruleVariationalNARMAXIn2PPNPPPP(ϕ :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_u :: Nothing,  
										 marg_x :: ProbabilityDistribution{Multivariate},
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})
	
	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalNARMAXIn3PPPNPPP(ϕ :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
									  	 marg_u :: ProbabilityDistribution{Univariate},	
										 marg_x :: Nothing,
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})

	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalNARMAXIn4PPPPNPP(ϕ :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate}, 
										 marg_x :: ProbabilityDistribution{Multivariate},
							  	      	 marg_z :: Nothing,
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})

	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalNARMAXIn5PPPPPNP(ϕ :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate},
										 marg_x :: ProbabilityDistribution{Multivariate},
							  	      	 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: Nothing,
										 marg_τ :: ProbabilityDistribution{Univariate})

	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalNARMAXIn6PPPPPPN(ϕ :: Function,
									 	 marg_y :: ProbabilityDistribution{Univariate},
										 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate},  
										 marg_x :: ProbabilityDistribution{Multivariate},
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
  										 marg_τ :: Nothing)

	# Extract moments of beliefs
	mθ,Vθ = unsafeMeanCov(marg_θ)
	my = unsafeMean(marg_y)
	mu = unsafeMean(marg_u)
	mx = unsafeMean(marg_x)
	mz = unsafeMean(marg_z)
	mr = unsafeMean(marg_r)

	# Basis expansion of moments
	ϕm = ϕ([mu; mx; mz; mr])

	# Update parameters
	α = 3/2.
	β = (my^2 - 2*my*(mθ'*ϕm) + (mθ'*ϕm)^2 + ϕm'*Vθ*ϕm)/2.

	# Set message
	return Message(Univariate, Gamma, a=α, b=β)
end


function collectNaiveVariationalNodeInbounds(node::NAutoRegressiveMovingAverageX, entry::ScheduleEntry)
	inbounds = Any[]

	# Push function to calling signature (ϕ needs to be defined in user scope)
	push!(inbounds, Dict{Symbol, Any}(:ϕ => node.ϕ, :keyword => false))

    target_to_marginal_entry = currentInferenceAlgorithm().target_to_marginal_entry

    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        if node_interface === entry.interface
            # Ignore marginal of outbound edge
            push!(inbounds, nothing)
        elseif (inbound_interface != nothing) && isa(inbound_interface.node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        else
            # Collect entry from marginal schedule
            push!(inbounds, target_to_marginal_entry[node_interface.edge.variable])
        end
    end

    return inbounds
end
