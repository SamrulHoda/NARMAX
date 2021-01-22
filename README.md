# NARMAX

Custom ForneyLab.jl factor node for a *N*onlinear *A*uto*R*egressive model with *M*oving *A*verage and e*X*ogenous input.

Caution: the node assumes white noise error terms.

### Usage

The node can be added by running

```julia
] dev git@github.com:biaslab/NARMAX.git
```

in the REPL and then be used by running

```julia 
using ForneyLab
using NARMAX
```

### Feedback

Questions and comments can be left in the [issues](https://github.com/biaslab/NARMAX/issues) tracker.
