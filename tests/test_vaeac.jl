import ProbAbEx as PAE
using Serialization, Random, Optimisers, Lux, Base, ColorTypes, NNlib
using FileIO, CairoMakie
using BSON: @save, @load


""" ================ Imputation and Sampling ================= """
function impute(ts::Lux.Training.TrainState, x, mask)
    ldim = getfield(ts.model, :ldim)
    ε = randn(Float32, ldim, size(x, 2))
    (logits, μq, logσq, μp, logσp), _ = Lux.apply(ts.model, (x, mask, ε), ts.parameters, ts.states)
    σ.(logits)
end

function sample_and_save_png(ts::Lux.Training.TrainState, x, mask; binary=true, path="vaeac_imputation.png")
    xhat = impute(ts, x, mask)
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

function sample_and_save(x, mask, ts; binary=true)
    ε = randn(Float32, getfield(ts.model, :ldim), size(x, 2))
    (logits, μq, logσq, μp, logσp), _ = Lux.apply(ts.model, (x, mask, ε), ts.parameters, ts.states)
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
ts = PAE.train_vaeac(epochs=20, lr=0.01f0, batch_size=150)



""" ================ Imputation and Sampling ================= """
x = PAE.load_binary_mnist_matrix()[:, 2]
mask = random_mask(5; D=784)


x_img = sample_and_save_png(ts2, x, mask; binary=true)
x_img2 = sample_and_save(x, mask, ts2, binary=true)



""" ================ Save and Load Model ================= """

save_vaeac_jls(ts, "models/mnist_vaeac_model_50.jls")
ts2 = load_vaeac_jls("models/mnist_vaeac_model_50.jls"; lr=0.001f0)


save_vaeac(ts, "models/mnist_vaeac_model_50.bson")
ts2 = load_vaeac("models/mnist_vaeac_model_50.bson"; lr=0.001f0)
