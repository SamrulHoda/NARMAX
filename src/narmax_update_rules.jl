import LinearAlgebra: I, tr, inv
import ForneyLab: unsafeCov, unsafeMean, unsafePrecision, VariateType,
				  collectNaiveVariationalNodeInbounds, assembleClamp!, ultimatePartner
# using Zygote
include("util.jl")

export ruleVariationalNARMAXOutNPPPPPP,
       ruleVariationalNARMAXIn1PNPPPPP,
       ruleVariationalNARMAXIn2PPNPPPP,
       ruleVariationalNARMAXIn3PPPNPPP,
	   ruleVariationalNARMAXIn4PPPPNPP,
	   ruleVariationalNARMAXIn5PPPPPNP,
	   ruleVariationalNARMAXIn6PPPPPPN

# # Autoregression orders
# order_out = 0
# order_inp = 0
# order_res = 0

# # Approximating point for Taylor series
# approxθ = 0.0

# function defineOrder(n_y::Int64, n_u::Int64, n_e::Int64)
# 	global order_out, order_inp, order_res

# 	# Autoregression order
#     order_out = n_y
# 	order_inp = n_u
# 	order_res = n_e
# end


function ruleVariationalNARMAXOutNPPPPPP(g :: Function,
									     marg_y :: Nothing,
									     marg_θ :: ProbabilityDistribution{Multivariate},
                                         marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate},
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})

	# Extract moments of beliefs
	mθ = unsafeMean(marg_θ)
	mx = unsafeMean(marg_x)
	mu = unsafeMean(marg_u)
	mz = unsafeMean(marg_z)
	mr = unsafeMean(marg_r)
	mτ = unsafeMean(marg_τ)

	# Set order
	# n_y = dims(marg_x)
	# n_u = dims(marg_z)
	# n_e = dims(marg_r)
	# defineOrder(n_y, n_u, n_e)

	# # Update approximating point
	# global approxθ = mθ

	# Evaluate f at mθ
	fθ = mθ'*g([mx; mu; mz; mr])

	# Set outgoing message
	return Message(Univariate, GaussianMeanPrecision, m=fθ, w=mτ)
end

function ruleVariationalNARMAXIn1PNPPPPP(g :: Function,
									     marg_y :: ProbabilityDistribution{Univariate},
                                         marg_θ :: Nothing,
									     marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate},
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})

    # Extract moments of beliefs
	my = unsafeMean(marg_y)
	mx = unsafeMean(marg_x)
	mu = unsafeMean(marg_u)
	mz = unsafeMean(marg_z)
	mr = unsafeMean(marg_r)
	mτ = unsafeMean(marg_τ)

	# Set order
	# n_y = dims(marg_x)
	# n_u = dims(marg_z)
	# n_e = dims(marg_r)
	# defineOrder(n_y, n_u, n_e)

	# Jacobian of f evaluated at mθ
	# Jθ = Zygote.gradient(g, approxθ, mx, mu, mz, mr)[1]
	Jθ = g([mx; mu; mz; mr])

	# Update parameters
	Φ = mτ*Jθ*Jθ'
	ϕ = mτ*my*Jθ

	# Regularize matrix to ensure invertibility
	# reg = 1e-8*Matrix{Float64}(I, size(Φ))

	# Update approximating point
	# global approxθ = inv(Φ + reg)*ϕ

	# Set message
    return Message(Multivariate, GaussianWeightedMeanPrecision, xi=ϕ, w=Φ)
end

function ruleVariationalNARMAXIn2PPNPPPP(g :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_x :: Nothing,
										 marg_u :: ProbabilityDistribution{Univariate},  
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})
	
	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalNARMAXIn3PPPNPPP(g :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
									  	 marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: Nothing,	
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})

	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalNARMAXIn4PPPPNPP(g :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate}, 
							  	      	 marg_z :: Nothing,
										 marg_r :: ProbabilityDistribution{Multivariate},
										 marg_τ :: ProbabilityDistribution{Univariate})

	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalNARMAXIn5PPPPPNP(g :: Function,
									  	 marg_y :: ProbabilityDistribution{Univariate},
									  	 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate},
							  	      	 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: Nothing,
										 marg_τ :: ProbabilityDistribution{Univariate})

	error("Variable is assumed to be observed")
    return Nothing
end

function ruleVariationalNARMAXIn6PPPPPPN(g :: Function,
									 	 marg_y :: ProbabilityDistribution{Univariate},
										 marg_θ :: ProbabilityDistribution{Multivariate},
										 marg_x :: ProbabilityDistribution{Multivariate},
										 marg_u :: ProbabilityDistribution{Univariate},  
										 marg_z :: ProbabilityDistribution{Multivariate},
										 marg_r :: ProbabilityDistribution{Multivariate},
  										 marg_τ :: Nothing)

	# Extract moments of beliefs
	my = unsafeMean(marg_y)
	mθ = unsafeMean(marg_θ)
	mx = unsafeMean(marg_x)
	mu = unsafeMean(marg_u)
	mz = unsafeMean(marg_z)
	mr = unsafeMean(marg_r)
	Vθ = unsafeCov(marg_θ)

	# Set order
	# n_y = dims(marg_x)
	# n_u = dims(marg_z)
	# n_e = dims(marg_r)
	# defineOrder(n_y, n_u, n_e)

	# Update approximating point
	# global approxθ = mθ

	# Gradient of f evaluated at mθ
	# Jθ = Zygote.gradient(g, mθ, mx, mu, mz, mr)[1]
	Jθ = g([mx; mu; mz; mr])

	# Evaluate f at mθ
	fθ = mθ'*Jθ

	# Update parameters
	a = 3/2.
	b = (my^2 - 2*my*fθ + fθ^2 + Jθ'*Vθ*Jθ)/2.

	# Set message
	return Message(Univariate, Gamma, a=a, b=b)
end


function collectNaiveVariationalNodeInbounds(node::NAutoRegressiveMovingAverageX, entry::ScheduleEntry)
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
