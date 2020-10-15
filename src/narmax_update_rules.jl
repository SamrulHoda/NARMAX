import LinearAlgebra: I, Hermitian, tr, inv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType,
				  collectNaiveVariationalNodeInbounds, assembleClamp!, ultimatePartner
using Zygote
include("util.jl")

export ruleVariationalNARMAXOutNPPPPP,
       ruleVariationalNARMAXIn1PNPPPP,
       ruleVariationalNARMAXIn2PPNPPP,
       ruleVariationalNARMAXIn3PPPNPP,
	   ruleVariationalNARMAXIn4PPPPNP,
	   ruleVariationalNARMAXIn5PPPPPN

# Autoregression orders
order_out = Nothing
order_inp = Nothing

# Approximating point for Taylor series
approxθ = 0.0

function defineOrder(M::Int64, N::Int64)
	global order_out, order_inp, approxθ

	# Autoregression order
    order_out = M
	order_inp = N
end


function ruleVariationalNARMAXOutNPPPPP(g :: Function,
									    marg_y :: Nothing,
									    marg_θ :: ProbabilityDistribution{Multivariate},
                                        marg_x :: ProbabilityDistribution{Multivariate},
                                        marg_z :: ProbabilityDistribution{Multivariate},
                                        marg_u :: ProbabilityDistribution{Univariate},
                                        marg_τ :: ProbabilityDistribution{Univariate})

	return Nothing
end

function ruleVariationalNARMAXIn1PNPPPP(g :: Function,
									    marg_y :: ProbabilityDistribution{Univariate},
                                        marg_θ :: Nothing,
									    marg_x :: ProbabilityDistribution{Multivariate},
                                        marg_z :: ProbabilityDistribution{Multivariate},
                                        marg_u :: ProbabilityDistribution{Univariate},
                                        marg_τ :: ProbabilityDistribution{Univariate})

    return Nothing
end

function ruleVariationalNARMAXIn2PPNPPP(g :: Function,
									  	marg_y :: ProbabilityDistribution{Univariate},
									  	marg_θ :: ProbabilityDistribution{Multivariate},
									  	marg_x :: Nothing,
							  	      	marg_z :: ProbabilityDistribution{Multivariate},
								      	marg_u :: ProbabilityDistribution{Univariate},
                                      	marg_τ :: ProbabilityDistribution{Univariate})

    return Nothing
end

function ruleVariationalNARMAXIn3PPPNPP(g :: Function,
									  	marg_y :: ProbabilityDistribution{Univariate},
									  	marg_θ :: ProbabilityDistribution{Multivariate},
									  	marg_x :: ProbabilityDistribution{Multivariate},
							  	      	marg_z :: Nothing,
								      	marg_u :: ProbabilityDistribution{Univariate},
                                      	marg_τ :: ProbabilityDistribution{Univariate})

    return Nothing
end

function ruleVariationalNARMAXIn4PPPPNP(g :: Function,
									  	marg_y :: ProbabilityDistribution{Univariate},
									  	marg_θ :: ProbabilityDistribution{Multivariate},
									  	marg_x :: ProbabilityDistribution{Multivariate},
							  	      	marg_z :: ProbabilityDistribution{Multivariate},
								      	marg_u :: Nothing,
                                      	marg_τ :: ProbabilityDistribution{Univariate})

    return Nothing
end

function ruleVariationalNARMAXIn5PPPPPN(g :: Function,
									  	marg_y :: ProbabilityDistribution{Univariate},
									  	marg_θ :: ProbabilityDistribution{Multivariate},
									  	marg_x :: ProbabilityDistribution{Multivariate},
							  	      	marg_z :: ProbabilityDistribution{Multivariate},
								      	marg_u :: ProbabilityDistribution{Univariate},
                                      	marg_τ :: Nothing)

    return Nothing
end


function collectNaiveVariationalNodeInbounds(node::NAutoregressiveMovingaverageX, entry::ScheduleEntry)
	inbounds = Any[]

	# Push function to calling signature (g needs to be defined in user scope)
	push!(inbounds, Dict{Symbol, Any}(:g => node.g, :keyword => false))

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
