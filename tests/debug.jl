# srun -p gpufast --gres=gpu:1  --mem=16000  --pty bash -i
# cd julia/Pkg/Subset_minimal_search/tests/
#  ~/.juliaup/bn/julia --project=.
using Revise
# using ProfileCanvas, BenchmarkTools
using ProbAbEx
import ProbAbEx as PAE
# using ProbAbEx.LinearAlgebra
using ProbAbEx.MLDatasets
using ProbAbEx.StaticBitSets
using ProbAbEx.TimerOutputs
using ProbAbEx.Serialization
using StatsBase: indicatormat
using CUDA
using Flux

# using PAE.Makie

const to = ProbAbEx.to


CUDA.has_cuda()
CUDA.device()

to_gpu = gpu
# to_gpu = cpu

""" Usual nn """
model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model_path = "models/binary_model.jls"
model = deserialize(model_path) |> to_gpu;

""" check if model on GPU """
is_on_gpu = all(p -> p isa CUDA.CuArray, Flux.params(model))
println(is_on_gpu)

""" nn for MILP search """
# nn = Chain
# (Dense(28^2, 28, relu), Dense(28,28, relu), Dense(28,10)) 
# nn = train_nn(nn, train_X_bin_neg, train_y, test_X_bin_neg, test_y)


""" Prepare data """
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = PAE.preprocess_binary(train_X)
test_X_binary = PAE.preprocess_binary(test_X)

train_X_bin_neg = PAE.preprocess_bin_neg(train_X_binary)
test_X_bin_neg = PAE.preprocess_bin_neg(test_X_binary)

train_y = PAE.onehot_labels(train_y)
test_y = PAE.onehot_labels(test_y)

""" Prepare image and label """
xₛ = train_X_bin_neg[:, 1] |> to_gpu
yₛ = argmax(train_y[:, 1])
sm = PAE.Subset_minimal(model, xₛ, yₛ)

function init_sbitset(n::Int, k = 0) 
    N = ceil(Int, n / 64)
    x = SBitSet{N, UInt64}()
    k == 0 && return(x)
    for i in rand(1:n, k)
        x = push(x, i)
    end
    x
end

sampler = UniformDistribution()
sampler = BernoulliMixture(to_gpu(deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls"))))
# sampler_path = "Subset_minimal_search/models/milan_centers.jls"
# sampler = BernoulliMixture(to_gpu(deserialize(sampler_path)))
#test
ii = init_sbitset(784)
# data_matrix = data_distribution(xₛ, ii, r, 100)

#first image
# println(data_matrix[:, 1])
#change 0 to -1 and 1 to 1
# println(2*data_matrix[:, 1] .- 1)
# plot_mnist_image(2*data_matrix[:, 20] .- 1)

""" Test MILP search """
# """ Test one subset(layer) backward/forward/beam search """ -- too slow
# threshold denotes the required precision of the subset
# solution_subset = one_subset_forward_search(sm, criterium_sdp; data_model=r, max_steps=50, threshold=0.5, num_samples=10, terminate_on_first_solution=false)
# solution_subset = one_subset_backward_search(sm, criterium_sdp; data_model=r, max_steps=50, threshold=0.5, num_samples=100, time_limit=60)
# solution_beam_subsets = one_subset_beam_search(sm, criterium_ep; data_model=r, threshold=0.5, beam_size=5, num_samples=100, time_limit=60)


""" Test search functions that fit one and all subsets"""
# Threshold denotes allowed error of the subset
# Valid_criterium is needed to distinguish with respect to what the validity of a subset is being checked, because it can be different, e.g. as with MILP 

reset_timer!(to)

#1. Initialization
II = (init_sbitset(784), nothing, nothing)

#  serialize("/home/pevnytom/tmp/subsets.jls", (;solutions = collect(solution_subsets), x = Vector(xₛ)))

samplers = (UniformDistribution(), UniformDistribution(), UniformDistribution())
# samplers = (BernoulliMixture(centers[:,1:784,:]), BernoulliMixture(centers[:,785:1040,:]), BernoulliMixture(centers[:,1041:end,:]))
ϵ = 0.9
#2. Search
""" Prepare image and label """
img_i = 1
xₛ = train_X_bin_neg[:, img_i] |> to_gpu
yₛ = argmax(model(xₛ))

# variant with triplets
# II = (init_sbitset(784), nothing, nothing)
# sm = Subset_minimal(model, xₛ, yₛ, (784, 256, 256))

# variant with just input
# II = init_sbitset(length(xₛ))
# sm = Subset_minimal(model, xₛ, yₛ)
sm = PAE.Subset_minimal(model, xₛ, yₛ, (784, 256, 256))
II = (init_sbitset(784), init_sbitset(256), init_sbitset(256))

# t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 100),  ShapleyHeuristic(sm, sampler, 100), refine_with_backward = false)
# t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, 10000),  ii -> heuristic_sdp(ii, sm, ϵ, sampler, 10000))
# t = @elapsed solution_subsets = forward_search(sm, II, ii -> isvalid_ep(ii, sm, ϵ, sampler, 10000),  ii -> heuristic_ep(ii, sm, ϵ, sampler, 10000))

# CUDA.@time forward_search(sm, II, ii -> isvalid_ep(ii, sm, ϵ, sampler, 10000),  ii -> heuristic_ep(ii, sm, ϵ, sampler, 10000), refine_with_backward = false)

# solutions = forward_search(sm, II, ii -> isvalid_sdp3(II, sm, ϵ, samplers, 1000),  ShapleyHeuristic(sm, samplers, 1000); terminate_on_first_solution=true, refine_with_backward = false)


# verification checks
function test_samplers()    
    # using Test
    xₛ = train_X_bin_neg[:, 1]

    ii = init_sbitset(784, 64)
    vii = collect(ii)
    cii = setdiff(1:length(xₛ), ii)

    sampler_gpu = BernoulliMixture(to_gpu(deserialize(joinpath("Subset_minimal_search", "models", "milan_centers.jls"))))
    sampler_cpu = BernoulliMixture(deserialize(joinpath("Subset_minimal_search", "models", "milan_centers.jls")))

    r_cpu = condition(sampler_cpu, cpu(xₛ), ii)
    r_gpu = condition(sampler_gpu, cu(xₛ), ii)

    for xx in [cpu(sample_all(r_cpu, 10_000)),cpu(sample_all(r_gpu, 10_000))]
        all(xx .!= 0)
        all(map(∈((-1,+1)), xx))
        all(xx[vii,:] .== xₛ[vii])
    end

    mean(xx[cii,:], dims =2 )
end

# test_samplers()

# CUDA.reclaim()