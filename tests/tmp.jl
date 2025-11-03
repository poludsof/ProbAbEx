using Revise
# import ProbAbEx as PAE
# # using ProbAbEx.TimerOutputs
# using StatsBase: indicatormat
using Serialization
using CUDA

@show CUDA.has_cuda() CUDA.functional()

using Flux
# using NNlib


CUDA.has_cuda()
CUDA.device()

to_gpu = gpu
# to_gpu = cpu

model = deserialize(joinpath(@__DIR__, "..", "models", "binary_model.jls")) |> to_gpu;

""" check if model on GPU """
is_on_gpu = all(p -> p isa CUDA.CuArray, Flux.params(model))
println("model is on gpu: ", is_on_gpu)

using ProbAbEx
import ProbAbEx as PAE
centers = deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls")) |> to_gpu;
sampler = BernoulliMixture(centers)

@show typeof(sampler)


using ProbAbEx.MLDatasets
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = PAE.preprocess_binary(train_X)
test_X_binary = PAE.preprocess_binary(test_X)

train_X_bin_neg = PAE.preprocess_bin_neg(train_X_binary)
test_X_bin_neg = PAE.preprocess_bin_neg(test_X_binary)

train_y = PAE.onehot_labels(train_y)
test_y = PAE.onehot_labels(test_y)

xₛ = train_X_bin_neg[:, 1] |> to_gpu;
yₛ = argmax(model(xₛ))
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

using ProbAbEx.StaticBitSets
II = init_sbitset(length(xₛ))
num_samples = 100
ϵ = 0.9

const to = ProbAbEx.to
include("/home/sofia/ProbAbEx/ext/CUDAExt.jl")
t = @elapsed solution_subsets = PAE.forward_search(sm, II, ii -> PAE.isvalid_sdp(ii, sm, ϵ, sampler, 100),  PAE.ShapleyHeuristic(sm, sampler, 100), refine_with_backward = false)

function  get_image(img_i)
    xₛ = train_X_bin_neg[:, img_i] |> to_gpu
    yₛ = argmax(model(xₛ))
    sm = PAE.Subset_minimal(to_gpu(model), xₛ, yₛ)
    xₛ, yₛ, sm
end

function run_experiment_forward(num_img, ϵ, num_samples)
    results = []
    img_i = 1
    successful_images = 0

    while successful_images < num_img

        xₛ, yₛ, sm = get_image(img_i)
        II = init_sbitset(length(xₛ))
        println("Image: $img_i, successful_images: $successful_images, ϵ: $ϵ, num_samples: $num_samples")

        time = @elapsed solution_subsets = PAE.forward_search(
            sm, II, ii -> PAE.isvalid_sdp(ii, sm, ϵ, sampler, num_samples),
            PAE.ShapleyHeuristic(sm, sampler, num_samples), 
            time_limit = 200,
            terminate_on_first_solution = true,
            refine_with_backward = false)

        CUDA.synchronize()

        if solution_subsets !== nothing && time <= 200
            subset_size = length(solution_subsets) > 0 ? length(solution_subsets) : 0
            push!(results, (image=img_i, ϵ=ϵ, num_samples=num_samples, time=time, subset_size=subset_size, solution=solution_subsets))
            successful_images += 1
        end

        CUDA.reclaim()
        img_i += 1
    end

    return results
end

function all_forward_results()
    results = []
    push!(results, run_experiment_forward(10, 0.9, 1000))
    println("Finished 1")
    push!(results, run_experiment_forward(10, 0.99, 1000))
    println("Finished 2")
    push!(results, run_experiment_forward(10, 0.9, 10000))
    println("Finished 3")
    push!(results, run_experiment_forward(10, 0.99, 10000))
    println("Finished 4")
    push!(results, run_experiment_forward(10, 0.9, 100000))
    println("Finished 5")
    push!(results, run_experiment_forward(10, 0.99, 100000))
    println("Finished 6")
    results
end

results = all_forward_results()


# compute averages
data_09_1000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.9 && getproperty(x, :num_samples) == 1000, vcat(results...)) 
data_099_1000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.99 && getproperty(x, :num_samples) == 1000, vcat(results...))
data_09_10000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.9 && getproperty(x, :num_samples) == 10000, vcat(results...))
data_099_10000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.99 && getproperty(x, :num_samples) == 10000, vcat(results...))
data_09_100000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.9 && getproperty(x, :num_samples) == 100000, vcat(results...))
data_099_100000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.99 && getproperty(x, :num_samples) == 100000, vcat(results...))

data = [data_09_1000; data_099_1000; data_09_10000; data_099_10000; data_09_100000; data_099_100000]
groups = Dict{Tuple{Float64,Int}, Vector{typeof(data[1])}}()
for x in data
    key = (getproperty(x, Symbol("ϵ")), getproperty(x, :num_samples))
    push!(get!(groups, key, Vector{typeof(x)}()), x)
end

using Statistics: mean
for ((eps, n), xs) in sort(collect(groups); by=first)
    mt = mean(getproperty.(xs, :time))
    ms = mean(getproperty.(xs, :subset_size))
    println("n: $n, ε: $eps -> mean time: $mt s, mean size: $ms")
end


#compute precision

function calculate_precision_new(img, solution_set)
    x = train_X_bin_neg[:, img] #|> to_gpu
    y = argmax(model(x))
    img_i = 0
    match_i = 0
    success = 0
    while match_i < 100 && img_i < 60000
        img_i += 1
        if img_i == img
            continue
        end
        xᵢ = train_X_bin_neg[:, img_i] #|> to_gpu
        yᵢ = argmax(model(xᵢ))

        match = sum(xᵢ[i] == x[i] for i in solution_set)
        if match == length(solution_set)
            match_i += 1
            success += (yᵢ == y)
        end
    end
    return success / match_i
end

function average_precision_new(results_data, epsilon, num_samples) 
    condition = (getproperty.(results_data, Symbol("ϵ")) .== epsilon) .& (getproperty.(results_data, :num_samples) .== num_samples)
    subset = results_data[condition]
    precision_list = Float64[]
    for row in subset
        img_i = getproperty(row, :image)
        solution = collect(getproperty(row, :solution))

        precision = calculate_precision_new(img_i, solution)
        push!(precision_list, precision)
    end
    precision_list = precision_list[precision_list .> 0]
    return mean(precision_list) * 100
end

model = cpu(model)
println("Precision n=1000  ε=0.9  = ", average_precision_new(results[1], 0.9, 1000))
println("Precision n=1000  ε=0.99 = ", average_precision_new(results[2], 0.99, 1000))
println("Precision n=10000 ε=0.9  = ", average_precision_new(results[3], 0.9, 10000))
println("Precision n=10000 ε=0.99 = ", average_precision_new(results[4], 0.99, 10000))
println("Precision n=100000 ε=0.9  = ", average_precision_new(results[5], 0.9, 100000))
println("Precision n=100000 ε=0.99 = ", average_precision_new(results[6], 0.99, 100000))