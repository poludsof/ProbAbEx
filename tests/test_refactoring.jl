
# include("../ext/CUDAExt.jl")

using Revise
using ProfileCanvas, BenchmarkTools
using CUDA
using ProbAbEx
import ProbAbEx as PAE
using ProbAbEx.Flux
using Serialization

to_gpu = gpu

model_path = "models/binary_model.jls"
model = deserialize(model_path) |> to_gpu;

sampler = UniformDistribution()
# sampler_path = "Subset_minimal_search/models/milan_centers.jls"
# sampler = BernoulliMixture(to_gpu(deserialize(sampler_path)))

using Subset_minimal_search.LinearAlgebra
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.TimerOutputs
const to = Subset_minimal_search.to

train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_X_bin_neg = preprocess_bin_neg(train_X_binary)
test_X_bin_neg = preprocess_bin_neg(test_X_binary)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)


function init_sbitset(n::Int, k = 0) 
    N = ceil(Int, n / 64)
    x = SBitSet{N, UInt64}()
    k == 0 && return(x)
    for i in rand(1:n, k)
        x = push(x, i)
    end
    x
end

function isvalid_sdp(ii::Tuple, sm, ϵ, sampler, num_samples, verbose = false)
    acc = criterium_sdp(sm.nn, sm.input, sm.output, ii[1], sampler, num_samples)
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ)
    acc > ϵ
end

function heuristic_sdp(ii::Tuple, sm, ϵ, sampler, num_samples, verbose = false)
    h = heuristic(sm, criterium_sdp, sdp_partial, ii, ϵ, sampler, num_samples)
    verbose && println("heuristic = ",h)
    h
end


# For tuple, we need to define appropriate heuristics which understand tuples

function isvalid_sdp(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    acc = criterium_sdp(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ)
    acc > ϵ
end

function heuristic_sdp(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    h = criterium_sdp(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    h = ϵ - h
    verbose && println("heuristic = ",h)
    (;hsum = h, hmax = h)
end

function isvalid_ep(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    acc = criterium_ep(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ)
    acc > ϵ
end

function heuristic_ep(ii::SBitSet, sm, ϵ, sampler, num_samples, verbose = false)
    h = criterium_ep(sm.nn, sm.input, sm.output, ii, sampler, num_samples)
    h = ϵ - h
    verbose && println("heuristic = ",h)
    (;hsum = h, hmax = h)
end

function shapley_heuristic(ii::SBitSet, sm, sampler, num_samples, verbose = false)
    r = condition(sampler, sm.input, ii)
    x = sample_all(r, num_samples)
    y = Flux.onecold(sm.nn(x)) .== sm.output

    # For us, the DAF statistic is a difference in errors when the feature is correct and wrong
    n = sum(x .== xₛ, dims = 2)
    # n₊ = sum(x .== 1, dims = 2)
    # n₋ = size(x,2) .- n₊

    # compute, how many +1 are correct
    _f(xᵢ, xₛᵢ, yᵢ) = xᵢ == xₛᵢ ? yᵢ : false
    c = sum(_f.(x, xₛ, y'), dims = 2)

    cpu(c ./ max.(1, n))
end

struct ShapleyHeuristic{S,SM}
    sm::SM
    sampler::S
    num_samples::Int
    verbose::Bool
end

function ShapleyHeuristic(sm, sampler, num_samples, verbose = false)
    ShapleyHeuristic(sampler, sm, num_samples, verbose)
end

(sp::ShapleyHeuristic)(ii) = heuristic_sdp(ii, sm, 0.99, sampler, 10000)

function Subset_minimal_search.expand_frwd(sm::Subset_minimal_search.Subset_minimal, stack, closed_list, ii::SBitSet, heuristic_fun::ShapleyHeuristic)
    sp = heuristic_fun
    acc = @timeit to "heuristic" shapley_heuristic(ii, sp.sm, sp.sampler, sp.num_samples, sp.verbose)
    for i in setdiff(1:sm.dims, ii)
        new_subset = push(ii, i)
        if new_subset ∉ closed_list
            new_error = 1 -acc[i]
            push!(stack, (new_error, new_error, new_subset))
        end
    end
    stack
end

is_on_gpu = any(x -> x isa CUDA.CuArray, Flux.params(model))











# TODO
#? forward and backward greedy search + dfs/bfs
#! milp choice (heuristic + criterium)
#! valid criterium (for milp)
#// sdp/ep choice
#// Attempt to write one search for all
#! ep_partial doesn't work
#// beam search for all
#// timeouts
#! terminate on the first valid subset
#// threshold of the error
#! implicative_subsets
#// forward/backward/beam add isvalid
#! distribution choice
#! check isvalid (threshold error and 0 <=)

#? если bacward имеет много решений в начале, хранить ли все? terminate on the first solution? 


# solution = forward_search_for_all(sm, (I3, I2, I1), ep, ep_partial, threshold_total_err=0.5, num_samples=100)
# reduced_solution = backward_reduction_for_all(sm, solution, sdp, sdp_partial, threshold=0.5, max_steps=100, num_samples=100)


# fix_inputs = collect(1:7)
# adversarial(model, sm.input, sm.output, fix_inputs)
