import ProbAbEx as PAE
using Serialization, Random, Optimisers, Lux, Base, ColorTypes, NNlib, MLDatasets, StaticBitSets
using FileIO, CairoMakie, Reactant
using BSON: @save, @load

""" ================ Imputation and Sampling ================= """
function impute(x, mask, model, ps, st)
    ldim = getfield(model, :ldim)
    ε = randn(Float32, ldim, size(x, 2))
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
    ax = Axis(fig[1, 1], title = "Masked MNIST Reconstruction", yreversed = true, aspect = DataAspect())
    image!(ax, color_image, interpolate = false)
    hidespines!(ax)
    hidedecorations!(ax)

    fig
end



""" ================ Save and Load Model ================= """
# BSON
function save_vaeac(ts::Lux.Training.TrainState, path::AbstractString)
    @save path model=ts.model ps=ts.parameters st=ts.states
end

function load_vaeac(path::AbstractString; lr=learning_rate)
    d = load(path)
    model = d[:model]
    ps = d[:ps]
    st = d[:st]
    Lux.Training.TrainState(model, ps, st, Optimisers.Adam(lr))
end

# JLS
function save_vaeac_jls(ts::Lux.Training.TrainState, path::AbstractString)
    open(path, "w") do io
        serialize(io, (model=ts.model, ps=ts.parameters, st=ts.states))
    end
end

function load_vaeac_jls(path::AbstractString; lr=learning_rate)
    data = open(deserialize, path)
    Lux.Training.TrainState(data.model, data.ps, data.st, Optimisers.Adam(lr))
end


""" ========= Create and train model ========= """
ts = PAE.train_vaeac(epochs=10, lr=0.001f0, batch_size=150)



""" ================ Imputation and Sampling ================= """

to_cpu(x) = x
to_cpu(x::AbstractArray) = Array(x)
to_cpu(nt::NamedTuple) = NamedTuple{keys(nt)}(map(to_cpu, values(nt)))
to_cpu(t::Tuple) = map(to_cpu, t)
# to_cpu(ts) = Lux.Training.TrainState(ts.model, to_cpu(ts.parameters), to_cpu(ts.states), ts.optimizer)

ps = to_cpu(ts.parameters)
st = to_cpu(ts.states)
model = ts.model

x = reshape(Float32.(PAE.load_binary_mnist_matrix()[:, 2]), :, 1)
mask = reshape(Float32.(random_mask(50; D=784)), :, 1)

x_img = sample_and_save_png(x, mask, model, ps, st; binary=true)
# x_img2 = sample_and_save(x, mask, model, ps, st, binary=true)


""" ================ Save and Load Model ================= """

# save_vaeac_jls(ts, "models/mnist_vaeac_model_tmp.jls")

ts2 = load_vaeac_jls("models/mnist_vaeac_model_50.jls"; lr=0.001f0)


# save_vaeac(ts, "models/mnist_vaeac_model_tmp.bson")

ts2 = load_vaeac("models/mnist_vaeac_model_50.bson"; lr=0.001f0)


ts3 = deserialize(joinpath(@__DIR__, "..", "models", "mnist_vaeac_model_50.jls"))

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


# Reactant.set_default_backend("gpu")
# dev = reactant_device()
using CUDA
dev = CUDA.has_cuda() ? cu : identity
CUDA.versioninfo()
@show CUDA.has_cuda()    # true
@show CUDA.functional()  # must be true
CUDA.device()

#! it needs Flux
using Flux
dev = gpu

model = dev(deserialize(joinpath(@__DIR__, "..", "models", "binary_model.jls")))
model = fmap(dev, model)
train_X_bin_neg, train_y, test_X_bin_neg, test_y = get_mnist_data()
xₛ = train_X_bin_neg[:, 1] #|> dev
yₛ =  argmax(train_y[:, 1])
sm = PAE.Subset_minimal(model, xₛ, yₛ)

II = init_sbitset(length(xₛ))
#or for backward
# II = init_full_sbitset(xₛ)

#? sampler
sampler = PAE.UniformDistribution()
sampler = PAE.BernoulliMixture(deserialize(joinpath(@__DIR__, "..", "models", "milan_centers.jls"))) #|> dev
sampler = PAE.VAEACSampler(deserialize("models/mnist_vaeac_model_20.jls")) |> dev

#? run search
#! add TIME
solution_subsets = PAE.beam_search(sm, II, ii -> PAE.isvalid_sdp(ii, sm, 0.3, sampler, 100), PAE.ShapleyHeuristic(sm, sampler, 100); beam_size=5, terminate_on_first_solution=true)
solution_subsets = PAE.forward_search(sm, II, ii -> PAE.isvalid_sdp(ii, sm, 0.9, sampler, 100), PAE.ShapleyHeuristic(sm, sampler, 100); refine_with_backward = false, terminate_on_first_solution=true)

# is_on_gpu = any(x -> x isa CUDA.CuArray, Flux.params(model))

# function model_on_gpu(m)
#     ps = Flux.params(m)
#     isempty(ps) && return false
#     all(p -> p isa CuArray, ps)
# end

# model_on_gpu(model)


#! todo
# solution_subsets = PAE.backward_search(sm, II, ii -> PAE.isvalid_sdp(ii, sm, 0.6, sampler, 1000), PAE.ShapleyHeuristic(sm, sampler, 1000))

