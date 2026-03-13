import ProbAbEx as PAE
using Serialization, Random, Optimisers, Lux, Base, NNlib, MLDatasets, StaticBitSets
using Reactant
using TimerOutputs
import Lux: σ


""" ================ Search ================= """
function get_mnist_data()
    train_X, train_y = MNIST(split=:train)[:]
    test_X, test_y = MNIST(split=:test)[:]

    train_X_binary = PAE.preprocess_binary(train_X)
    test_X_binary = PAE.preprocess_binary(test_X)

    train_X_bin_neg = PAE.preprocess_bin_neg(train_X_binary)
    test_X_bin_neg = PAE.preprocess_bin_neg(test_X_binary)

    train_y = PAE.onehot_labels(train_y)
    test_y = PAE.onehot_labels(test_y)

    return train_X_binary, train_y, test_X_binary, test_y
end

function init_sbitset(n::Int, k=0)
    N = ceil(Int, n / 64)
    x = SBitSet{N, UInt64}()
    k == 0 && return x

    for i in randperm(n)[1:k]
        x = push(x, i)
    end
    x
end

function init_full_sbitset(xₛ)
    II = SBitSet{13, UInt64}(collect(1:length(xₛ)))
    II
end

include("/home/sofia/ProbAbEx/ext/ReactantExt.jl")

Reactant.set_default_backend("gpu")
dev = reactant_device(; force=true)

model_cls, ps_cls, st_cls = deserialize(joinpath(@__DIR__, "..", "models", "mnist_conv_model.jls"))
ps_dev = ps_cls |> dev
st_dev = Lux.testmode(st_cls) |> dev

infer(model, x, ps, st) = first(Lux.apply(model, x, ps, st))

number_of_samples = 100000
batch_size = 1000

x0 = zeros(Float32, 28, 28, 1, batch_size) |> dev
compiled_infer = @compile infer(model_cls, x0, ps_dev, st_dev)
nn(x) = compiled_infer(model_cls, x, ps_dev, st_dev)

x1 = zeros(Float32, 28, 28, 1, 1) |> dev
compiled_infer_1 = @compile infer(model_cls, x1, ps_dev, st_dev)
nn1(x) = compiled_infer_1(model_cls, x, ps_dev, st_dev)

train_X_binary, train_y, test_X_binary, test_y = get_mnist_data()
xₛ = Float32.(train_X_binary[:, 2])
x_matrix = reshape(xₛ, 28, 28, 1, 1) |> dev
yₛ =  argmax(train_y[:, 2])
sm = PAE.Subset_minimal(nn, xₛ, yₛ)
println("sm.output: ", sm.output)

## checking if it works on GPU and just to see the predicted label
logits = Array(vec(nn1(x_matrix)))
pred_digit = argmax(logits)
println("Predicted label: ", pred_digit, " True label: ", yₛ)

# #? sampler
sampler = PAE.load_vaeac_sampler(joinpath(@__DIR__, "..", "models", "use_this_vaeac.jls"), dev)

dev = sampler.dev
ldim = sampler.model.ldim

# number_of_samples = 1000 #! defined above

# x01_B0 = zeros(Float32, 28, 28, 1, number_of_samples) |> dev
# m_B0   = zeros(Float32, 28, 28, 1, number_of_samples) |> dev
# ε0     = zeros(Float32, ldim, number_of_samples) |> dev
# u0     = zeros(Float32, 28, 28, 1, number_of_samples) |> dev

# compiled_sample = Reactant.@compile PAE.sample_all_core(
#     sampler.model.prior,
#     sampler.model.decoder,
#     ldim,
#     x01_B0,
#     m_B0,
#     ε0,
#     u0,
#     sampler.ps.prior,
#     sampler.st.prior,
#     sampler.ps.decoder,
#     sampler.st.decoder,
#     true
# )

compiled_acc = PAE.build_compiled_accuracy(sampler, model_cls, ps_dev, st_dev, dev, batch_size)
# compiled_acc = PAE.build_compiled_accuracy(sampler, model_cls, ps_dev, st_dev, dev, number_of_samples) #! not working with 100k samples

# using BenchmarkTools
II = init_sbitset(length(xₛ), 1)
println("length(II): ", length(II))
# r = PAE.condition(sampler, xₛ, II)
# x = PAE.sample_all(r, number_of_samples)
# @benchmark PAE.sample_all_compiled(r, compiled_sample, number_of_samples)
# @show size(x)

#! with batch size
acc = PAE.accuracy_sdp_batched(II, sm, sampler, number_of_samples, compiled_acc, model_cls, ps_dev, st_dev; batch_size=batch_size)


#! not with batch size
# acc = PAE.accuracy_sdp(II, sm, sampler, number_of_samples, compiled_acc, model_cls, ps_dev, st_dev; verbose=true) #! test accuracy_sdp + sample on GPU
# o = PAE.isvalid_sdp(II, sm, 0.9, sampler, number_of_samples, compiled_acc, model_cls, ps_dev, st_dev; verbose=true)
# h = PAE.heuristic_sdp(II, sm, 0.9, sampler, number_of_samples, compiled_acc, model_cls, ps_dev, st_dev; verbose=true)


##? run search
#! add TIME
# println("Fields of sm.nn: ", typeof(sm.nn))

to = TimerOutput()

# @timeit to "forward_search" begin 
#     solution_subsets = PAE.forward_search(sm, II, ii -> PAE.isvalid_sdp(ii, sm, 0.8, sampler, number_of_samples, compiled_acc, model_cls, ps_dev, st_dev), PAE.ShapleyHeuristic(sm, sampler, number_of_samples, compiled_acc, model_cls, ps_dev, st_dev); refine_with_backward = false, terminate_on_first_solution=true)
# end

@timeit to "accuracy and sample" begin
    acc = PAE.accuracy_sdp_batched(II, sm, sampler, number_of_samples, compiled_acc, model_cls, ps_dev, st_dev; batch_size=batch_size)
end
show(to)

## Sampling and visualization functions

using ColorTypes, FileIO
function save_samples_grid(x::AbstractArray{<:Real,3};
                           filename::AbstractString="samples_grid.png",
                           ncols::Int=5,
                           pad::Int=2)

    h, w, n = size(x)
    ncols = min(ncols, n)
    nrows = cld(n, ncols)

    H = nrows*h + (nrows+1)*pad
    W = ncols*w + (ncols+1)*pad

    out = fill(ColorTypes.RGB{Float32}(0,0,0), H, W)

    for k in 1:n
        r = (k-1) ÷ ncols
        c = (k-1) % ncols
        y0 = r*h + (r+1)*pad + 1
        x0 = c*w + (c+1)*pad + 1
        img = clamp.(Float32.(x[:, :, k]), 0f0, 1f0)
        out[y0:y0+h-1, x0:x0+w-1] .= RGB{Float32}.(img, img, img)
    end

    save(filename, out)
    filename
end
# save_samples_grid(x; filename="vaeac_samples.png", ncols=5)


# const to = TimerOutput()
# reset_timer!(to)
# PAE.accuracy_sdp(II, sm, sampler, 100)   # warmup
# reset_timer!(to)
# for _ in 1:5
#     PAE.accuracy_sdp(II, sm, sampler, 100)
# end
# show(Main.to)




function  get_image(img_i)
    xₛ = train_X_bin_neg[:, img_i] |> dev
    yₛ = argmax(model(xₛ))
    sm = PAE.Subset_minimal(model, xₛ, yₛ)
    xₛ, yₛ, sm
end

function vaeac_run_experiment_forward(num_img, ϵ, num_samples; num_samples_heu=num_samples)
    results = []
    img_i = 1
    successful_images = 0

    while successful_images < num_img

        xₛ, yₛ, sm = get_image(img_i)
        II = init_sbitset(length(xₛ))
        println("Image: $img_i, successful_images: $successful_images, ϵ: $ϵ, num_samples: $num_samples")

        
        time = @elapsed solution_subsets = PAE.forward_search(
            sm, II, ii -> PAE.isvalid_sdp(ii, sm, ϵ, sampler, num_samples),
            PAE.ShapleyHeuristic(sm, sampler, num_samples_heu), 
            time_limit = 300,
            terminate_on_first_solution = true,
            refine_with_backward = false)

        # CUDA.synchronize()

        if solution_subsets !== nothing && length(solution_subsets) != 0 && time <= 350
            subset_size = length(solution_subsets) > 0 ? length(solution_subsets) : 0
            push!(results, (image=img_i, ϵ=ϵ, num_samples=num_samples, time=time, subset_size=subset_size, solution=solution_subsets))
            successful_images += 1
        end

        # CUDA.reclaim()
        img_i += 1
    end

    return results
end

function all_forward_results()
    results = []
    # push!(results, vaeac_run_experiment_forward(10, 0.9, 1000))
    # println("Finished 1")
    # push!(results, vaeac_run_experiment_forward(10, 0.99, 1000))
    # println("Finished 2")
    push!(results, vaeac_run_experiment_forward(10, 0.9, 10000))
    println("Finished 3")
    push!(results, vaeac_run_experiment_forward(10, 0.99, 10000))
    println("Finished 4")
    # push!(results, vaeac_run_experiment_forward(10, 0.9, 100000, 10000))
    # println("Finished 5")
    # push!(results, vaeac_run_experiment_forward(10, 0.99, 100000, 10000))
    # println("Finished 6")
    results
end

# results_1000 = all_forward_results()
# results_10000 = all_forward_results()
# results_100000 = all_forward_results()

# # compute mean time and size
# data_09_1000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.9 && getproperty(x, :num_samples) == 1000, vcat(results_1000...)) 
# data_099_1000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.99 && getproperty(x, :num_samples) == 1000, vcat(results_1000...))
# data_09_10000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.9 && getproperty(x, :num_samples) == 10000, vcat(results_10000...))
# data_099_10000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.99 && getproperty(x, :num_samples) == 10000, vcat(results_10000...))
# data_09_100000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.9 && getproperty(x, :num_samples) == 100000, vcat(results_100000_1...))
# data_099_100000 = filter(x -> getproperty(x, Symbol("ϵ")) == 0.99 && getproperty(x, :num_samples) == 100000, vcat(results_100000_2...))

# data = [data_09_10000; data_099_10000] #; data_09_10000; data_099_10000] #; data_09_100000; data_099_100000]
# groups = Dict{Tuple{Float64,Int}, Vector{typeof(data[1])}}()
# for x in data
#     key = (getproperty(x, Symbol("ϵ")), getproperty(x, :num_samples))
#     push!(get!(groups, key, Vector{typeof(x)}()), x)
# end

# groups = Dict{Tuple{Float64,Int}, Vector{Any}}()
# for x in data_099_100000
#     key = (getproperty(x, Symbol("ϵ")), getproperty(x, :num_samples))
#     push!(get!(groups, key, Any[]), x)
# end

# using Statistics: mean
# for ((eps, n), xs) in sort(collect(groups); by=first)
#     mt = mean(getproperty.(xs, :time))
#     ms = mean(getproperty.(xs, :subset_size))
#     println("n: $n, ε: $eps -> mean time: $mt s, mean size: $ms")
# end


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
        # println("Image: $img_i, Solution: ", solution)

        if solution === nothing || length(solution) == 0
            continue
        end
        precision = calculate_precision_new(img_i, solution)
        push!(precision_list, precision)
    end
    precision_list = precision_list[precision_list .> 0]
    return mean(precision_list) * 100
end


# model = cpu(model)
# println("Precision n=1000  ε=0.9  = ", average_precision_new(results_1000[1], 0.9, 1000))
# println("Precision n=1000  ε=0.99 = ", average_precision_new(results_1000[2], 0.99, 1000))
# println("Precision n=10000 ε=0.9  = ", average_precision_new(results_10000[1], 0.9, 10000))
# println("Precision n=10000 ε=0.99 = ", average_precision_new(results_10000[2], 0.99, 10000))
# # println("Precision n=100000 ε=0.9  = ", average_precision_new(results[5], 0.9, 100000))
# println("Precision n=100000 ε=0.99 = ", average_precision_new(results[6], 0.99, 100000))


# model = cpu(model)
# solution_subsets
# img_i = 2
# calculate_precision_new(img_i, collect(solution_subsets))

# model = dev(model)
# results = all_forward_results()

# model = cpu(model)
# println("Precision n=100000  ε=0.9  = ", average_precision_new(results_100000_2[1], 0.99, 100000))



# Print few examples of solutions on images
function visualize_image_with_subset(image_idx, subset::Array, train_X_bin_neg)
    x = train_X_bin_neg[:, image_idx]
    mask = copy(x)
    mask[subset] .= 2
    
    reshaped_image = reshape(mask, 28, 28)
    
    img = Array{RGB{Float32}}(undef, 28, 28)
    
    for i in 1:28
        for j in 1:28
            # Checking if the pixel belongs to the subset
            if reshaped_image[i, j] == 2
                if x[(j-1)*28 + i, 1] == 1
                    println("Pixel in subset: ", (i, j))
                    img[i, j] = RGB{Float32}(1, 0, 0)
                else
                    println("Pixel in subset (not activated): ", (i, j))
                    img[i, j] = RGB{Float32}(0.5, 0, 0)
                end
            else
                gray_val = clamp(reshaped_image[i, j], 0, 1)
                img[i, j] = RGB{Float32}(gray_val, gray_val, gray_val)  # Grayscale for others
            end
        end
    end

    fig = Figure(resolution = (600, 600), backgroundcolor = :black)
    ax = Axis(fig[1, 1];
        aspect = DataAspect(),
        xgridvisible = false, ygridvisible = false,
        xticksvisible = false, yticksvisible = false,
        xticklabelsvisible = false, yticklabelsvisible = false,
        backgroundcolor = :black,
    )
    image!(ax, 0..28, 0..28, img; interpolate = false)
    for k in 0:28
        hlines!(ax, k; color = :gray20, linewidth = 2)
        vlines!(ax, k; color = :gray20, linewidth = 2)
    end

    xlims!(ax, 0, 28)
    ylims!(ax, 28, 0)

    save("vaeac_image_$(image_idx)_with_subset.png", fig)
    display(reverse(Base.rotr90(img), dims=2))
end



# imagw_idx = 1
# subset = [189,405,412,552,594,680]
# visualize_image_with_subset(imagw_idx, subset, train_X_bin_neg)

# julia> results_100000_2
#  Any[(image = 1,{189,405,412,552,594,680,,}),
# (image = 2,}{246,387,512,631,}), 
# (image = 3,}{129,202,286,456,458,523,625,632,688,692,}),
#  (image = 4 {265,296,325,489,544,682,687,}), 
#  (image = 6, {300,491,556,594,595,779,}), 
#  (image = 8, {189,302,372,375,379,439,594,633,651,}), 