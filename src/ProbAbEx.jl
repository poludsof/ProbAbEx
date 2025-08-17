module ProbAbEx

using CUDA
using Lux
using Lux: Chain, Dense, relu
# using Flux
using JuMP
using HiGHS
using LinearAlgebra
using MLDatasets
using Base.Iterators: partition
using Statistics: mean
using StaticBitSets
using TimerOutputs
using Optimisers
using MLUtils: DataLoader
using CairoMakie
using Makie
using Makie.Colors
using Serialization
using DataStructures
using Distributions
using Random
const to = TimerOutput()

struct Subset_minimal{NN, I, O, ID}
    nn::NN
    input::I
    output::O
    dims::ID
end

Subset_minimal(nn, input, output) = Subset_minimal(nn, input, output, length(input))
Subset_minimal(nn, input) = Subset_minimal(nn, input, nn(input))

include("mnist_training.jl")
include("plots.jl")
# include("milp.jl")
include("criterium.jl")
include("forward_search.jl")
include("backward_search.jl")
include("beam_search.jl")
include("dataset_prep.jl")
include("heuristic.jl")
include("utilities.jl")
include("heuristics_criteria.jl")
include("samplers/uniform_sampler.jl")
export UniformDistribution
include("samplers/mixture_sampler.jl")
export BernoulliMixture
export BatchHeuristic
include("samplers/VAEAC_sampler.jl")

# export one_subset_backward_search, one_subset_forward_search, one_subset_beam_search
# export preprocess_binary, preprocess_bin_neg, prepare_data
# export forward_search, beam_search, backward_search
# export criterium_sdp, criterium_ep, sdp_partial, ep_partial
# export plot_mnist_image
# export accuracy_sdp3, batch_heuristic3, isvalid_sdp3, restrict_output, expand_frwd
# export condition

end
