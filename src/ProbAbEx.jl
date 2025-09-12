module ProbAbEx

# using CUDA
using Lux
# using Flux
# using JuMP
# using HiGHS
# using LinearAlgebra
using MLDatasets
using StatsBase
using MLUtils
using Base.Iterators: partition
using Statistics
using StaticBitSets
using Base
# using TimerOutputs
using Optimisers
# using CairoMakie
# using Makie
# using Makie.Colors
using Serialization
# using DataStructures
# using Distributions
using Random
# using ADTypes
# using FileIO
# using ImageTransformations
# using ImageCore
# using NNlib: sigmoid
using Reactant
# using Reactant_jll
# using Functors
using Enzyme
using Revise

# const to = TimerOutput()

struct Subset_minimal{NN, I, O, ID}
    nn::NN
    input::I
    output::O
    dims::ID
end

Subset_minimal(nn, input, output) = Subset_minimal(nn, input, output, length(input))
Subset_minimal(nn, input) = Subset_minimal(nn, input, nn(input))

# include("mnist_training.jl")
# include("plots.jl")
# include("milp.jl")
# include("criterium.jl")

# include("forward_search.jl")
# include("backward_search.jl")
# include("beam_search.jl")
# include("dataset_prep.jl")
# include("heuristic.jl")
# include("utilities.jl")
# include("heuristics_criteria.jl")

export UniformDistribution
export BernoulliMixture
export BatchHeuristic
include("samplers/uniform_sampler.jl")
include("samplers/mixture_sampler.jl")
include("samplers/VAEAC_training.jl")
include("samplers/VAEAC_sampler.jl")

# include("datasets/celeba.jl")

end
