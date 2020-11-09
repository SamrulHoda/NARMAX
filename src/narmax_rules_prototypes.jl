@naiveVariationalRule(:node_type     => NAutoregressiveMovingAverageX,
                      :outbound_type => Message{GaussianMeanPrecision},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNARMAXOutNPPPPPP)

@naiveVariationalRule(:node_type     => NAutoregressiveMovingAverageX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNARMAXIn1PNPPPPP)

@naiveVariationalRule(:node_type     => NAutoregressiveMovingAverageX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNARMAXIn2PPNPPPP)

@naiveVariationalRule(:node_type     => NAutoregressiveMovingAverageX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNARMAXIn3PPPNPPP)

@naiveVariationalRule(:node_type     => NAutoregressiveMovingAverageX,
                      :outbound_type => Message{GaussianWeightedMeanPrecision},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalNARMAXIn4PPPPNPP)

@naiveVariationalRule(:node_type     => NAutoregressiveMovingAverageX,
                      :outbound_type => Message{Gamma},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalNARMAXIn5PPPPPNP)

@naiveVariationalRule(:node_type     => NAutoregressiveMovingAverageX,
                      :outbound_type => Message{Gamma},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalNARMAXIn5PPPPPPN)