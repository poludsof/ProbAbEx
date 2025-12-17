using ColorTypes: RGB
using ColorTypes: Gray, heatmap!
using CairoMakie: Figure, Axis, image!, hidespines!, hidedecorations!, save
import ProbAbEx as PAE
using Serialization, Random, Optimisers, Lux, Base, NNlib, MLDatasets, StaticBitSets
using Reactant
import Lux: σ

""" ================ Imputation and Sampling ================= """
function impute(x, mask, model, ps, st)
    ldim = getfield(model, :ldim)
    ε = randn(Float32, ldim, size(x, 2))
    # st = Lux.testmode(st)
    (logits, _, _, _, _), _ = Lux.apply(model, (x, mask, ε), ps, st)
    σ.(logits)
end

function sample_and_save_png(x, mask, model, ps, st; binary=true, path="vaeac_imputation.png")
    xhat = impute(x, mask, model, ps, st)
    g = reshape(xhat[:, 1], 28, 28)
    m = reshape(mask[:, 1], 28, 28)
    if binary
        g .= ifelse.(g .> 0.5f0, 1f0, 0f0)
    end
    img = Array{RGB{Float32}}(undef, 28, 28)

    for i in 1:28, j in 1:28
        if m[i, j] == 1
            img[i, j] = RGB{Float32}(0.5, 0, 0)
            if x[(j-1)*28 + i, 1] == 1
                img[i, j] = RGB{Float32}(1, 0, 0)
            end
        else
            gray_val = clamp(g[i, j], 0, 1)
            img[i, j] = RGB{Float32}(gray_val, gray_val, gray_val)
        end
    end
    display(reverse(Base.rotr90(img), dims=2))
end

random_mask(n; D=784, rng=Random.default_rng()) = (m = falses(D); m[view(randperm(rng, D), 1:n)] .= true; m )

function sample_and_save(x, mask, model, ps, st; binary=true)
    ε = randn(Float32, getfield(model, :ldim), size(x, 2))
    st = Lux.testmode(st)
    (logits, _, _, _, _), _ = Lux.apply(model, (x, mask, ε), ps, st)
    x_hat = σ.(logits)

    grayscale_image = x_hat[:, 1] isa AbstractVector ? reshape(x_hat[:, 1], 28, 28) : x_hat
    mask_image = mask[:, 1] isa AbstractVector ? reshape(mask[:, 1], 28, 28) : mask

    if binary
        grayscale_image .= ifelse.(grayscale_image .> 0.5, 1f0, 0f0)
    end

    color_image = Array{RGB{Float32}}(undef, 28, 28)

    for i in 1:28, j in 1:28
        if mask_image[i, j] == 1
            color_image[i, j] = RGB{Float32}(0.5, 0, 0)
            if x[(j-1)*28 + i, 1] == 1
                color_image[i, j] = RGB{Float32}(1, 0, 0)
            end
        else
            gray_val = clamp(grayscale_image[i, j], 0, 1)
            color_image[i, j] = RGB{Float32}(gray_val, gray_val, gray_val)
        end
    end

    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1], title = "Masked MNIST Reconstruction", yreversed = true) #, aspect = DataAspect())
    image!(ax, color_image, interpolate = false)
    hidespines!(ax)
    hidedecorations!(ax)

    save("vaeac_image.png", fig)

end

""" ================ Save and Load Model ================= """
# JLS
function save_vaeac_model(filename, model, ps, st)
    cdev = cpu_device()
    ps_cpu = cdev(ps)
    st_cpu = cdev(st)
    print("Saving model to $filename ... ")
    open(filename, "w") do io
        serialize(io, (model, ps_cpu, st_cpu))
    end
    println("Done!")
end

""" ========= Create and train model ========= """
ts = PAE.train_vaeac(epochs=50, lr=0.001f0, batch_size=100)

# model, ps, st = deserialize(joinpath(@__DIR__, "..", "models", "mnist_vaeac_conv_model.jls"))
# model, ps, st = deserialize("models/mnist_vaeac_conv_model.jls")
# # print fields of ts: model, parameters, states, optimizer
# println("Fields of the __: ", fieldnames(typeof(ts)))


""" ================ Imputation and Sampling ================= """
to_cpu(x) = x
to_cpu(x::AbstractArray) = Array(x)
to_cpu(nt::NamedTuple) = NamedTuple{keys(nt)}(map(to_cpu, values(nt)))
to_cpu(t::Tuple) = map(to_cpu, t)
# to_cpu(ts) = Lux.Training.TrainState(ts.model, to_cpu(ts.parameters), to_cpu(ts.states), ts.optimizer)

ps = to_cpu(ts.parameters)
st = to_cpu(ts.states)
model = ts.model

x_cpu = to_cpu(reshape(Float32.(PAE.load_binary_mnist_matrix()[:, 1]), :, 1))
mask = reshape(Float32.(random_mask(50; D=784)), :, 1)

# # x_img = sample_and_save_png(x_cpu, mask, model, ps, st; binary=true)
# x_img2 = sample_and_save(x_cpu, mask, model, ps, st, binary=true)

""" ================ Save and Load Model ================= """

# save_vaeac_model("models/mnist_vaeac_conv_model.jls", ts.model, ts.parameters, ts.states)

# ts2 = load_vaeac_jls("models/mnist_vaeac_model_50.jls"; lr=0.001f0)

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

    return train_X_bin_neg, train_y, test_X_bin_neg, test_y
end

function init_sbitset(n::Int, k = 0) 
    N = ceil(Int, n / 64)
    x = SBitSet{N, UInt64}()
    k == 0 && return(x)
    for i in rand(1:n, k)
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
dev = reactant_device()

model = deserialize(joinpath(@__DIR__, "..", "models", "mnist_conv_model.jls")) |> dev
# # is_on_gpu = all(p -> p isa CUDA.CuArray, Flux.params(model))

train_X_bin_neg, train_y, test_X_bin_neg, test_y = get_mnist_data()
xₛ = train_X_bin_neg[:, 2] |> dev
yₛ =  argmax(train_y[:, 2])
sm = PAE.Subset_minimal(model, xₛ, yₛ)

II = init_sbitset(length(xₛ))
# #or for backward
# # II = init_full_sbitset(xₛ)

# #? sampler
# sampler = PAE.UniformDistribution()
sampler = PAE.BernoulliMixture(dev(deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls"))))
# vaeac, ps, st = deserialize(joinpath(@__DIR__, "..", "models", "mnist_vaeac_conv_model.jls")) |> dev
# vaeac = vaeac |> dev
# ps    = ps    |> dev
# st    = st    |> dev
# sampler = PAE.VAEACSampler(vaeac, ps, st)

#? run search
#! add TIME
println("Fields of sm.nn: ", fieldnames(typeof(sm.nn)))
# solution_subsets = PAE.forward_search(sm, II, ii -> PAE.isvalid_sdp(ii, sm, 0.3, sampler, 100), PAE.ShapleyHeuristic(sm, sampler, 100); refine_with_backward = false, terminate_on_first_solution=true)


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