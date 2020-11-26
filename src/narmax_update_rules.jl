import LinearAlgebra: I, Hermitian, tr, inv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType,
				  collectNaiveVariationalNodeInbounds, assembleClamp!, ultimatePartner
using Zygote
include("util.jl")

export ruleVariationalNARMAXOutNPPPPPP,
       ruleVariationalNARMAXIn1PNPPPPP,
       ruleVariationalNARMAXIn2PPNPPPP,
       ruleVariationalNARMAXIn3PPPNPPP,
	   ruleVariationalNARMAXIn4PPPPNPP,
	   ruleVariationalNARMAXIn5PPPPPNP,
	   ruleVariationalNARMAXIn6PPPPPPN

# Autoregression orders
order_out = Nothing
order_inp = Nothing
order_res = Nothing

# Approximating point for Taylor series
approxθ = 0.0

function defineOrder(n_y::Int64, n_u::Int64, n_e::Int64)
	global order_out, order_inp, order_prn, approxθ

	# Autoregression order
    order_out = n_y
	order_inp = n_u
	order_res = n_e
end


function ruleVariationalNARMAXOutNPPPPPP(g :: Function,
									     marg_y :: Nothing,
									     marg_θ :: ProbabilityDistribution{Multivariate},
                                         marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate},
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate},
										 marg_γ :: ProbabilityDistribution{Multivariate})

	return Nothing
end

function ruleVariationalNARMAXIn1PNPPPPP(g :: Function,
									     marg_y :: ProbabilityDistribution{Univariate},
                                         marg_θ :: Nothing,
									     marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate},
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate},
										 marg_γ :: ProbabilityDistribution{Multivariate})

    return Nothing
end

function ruleVariationalNARMAXIn2PPNPPPP(g :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_x :: Nothing,
										 marg_u :: ProbabilityDistribution{Univariate},  
							  	      	 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate},
										 marg_u :: ProbabilityDistribution{Multivariate})

    return Nothing
end

function ruleVariationalNARMAXIn3PPPNPPP(g :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
									  	 marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: Nothing,	
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate},
										 marg_γ :: ProbabilityDistribution{Multivariate})

    return Nothing
end

function ruleVariationalNARMAXIn4PPPPNPP(g :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate}, 
							  	      	 marg_z :: Nothing,
										 marg_τ :: ProbabilityDistribution{Univariate},
										 marg_γ :: ProbabilityDistribution{Multivariate})

    return Nothing
end

function ruleVariationalNARMAXIn5PPPPPNP(g :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
									  	 marg_x :: ProbabilityDistribution{Multivariate},
							  	      	 marg_z :: ProbabilityDistribution{Multivariate},
								      	 marg_u :: ProbabilityDistribution{Univariate},
										 marg_τ :: Nothing,
										 marg_γ :: ProbabilityDistribution{Multivariate})

    return Nothing
end

function ruleVariationalNARMAXIn6PPPPPPN(g :: Function,
									 	 marg_y :: ProbabilityDistribution{Univariate},
										 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_x :: ProbabilityDistribution{Multivariate},
	  									 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate},
										 marg_τ :: ProbabilityDistribution{Univariate},
  										 marg_γ :: Nothing)

	return Nothing
end

function collectNaiveVariationalNodeInbounds(node::NAutoregressiveMovingAverageX, entry::ScheduleEntry)
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
