
# import ProbAbEx as PAE
# import Lux: initialparameters, initialstates
# todo: GPU, Reactant.jl
# todo: IWAE

""" ========= Hyperparameters ========= """
const input_dim = 28 * 28
const latent_dim = 20
const hidden_dim = 400

struct VAEAC{P,Q,D} <: Lux.AbstractLuxLayer
    proposal::P
    prior::Q
    decoder::D
    idim::Int
    ldim::Int
end

function VAEAC(idim::Int, ldim::Int, h::Int)
    proposal = Lux.Chain(
        Lux.Dense(2idim => h, relu),
        Lux.Dense(h => h, relu),
        Lux.Dense(h => 2ldim)
    )
    prior = Lux.Chain(
        Lux.Dense(idim => h, relu),
        Lux.Dense(h => h, relu),
        Lux.Dense(h => 2ldim)
    )
    decoder = Lux.Chain(
        Lux.Dense(ldim + idim => h, relu),
        Lux.Dense(h => h, relu),
        Lux.Dense(h => idim)
    )
    return VAEAC(proposal, prior, decoder, idim, ldim)
end

Lux.initialparameters(rng::AbstractRNG, m::VAEAC) = (
    proposal = Lux.initialparameters(rng, m.proposal),
    prior    = Lux.initialparameters(rng, m.prior),
    decoder  = Lux.initialparameters(rng, m.decoder),
)

Lux.initialstates(rng::AbstractRNG, m::VAEAC) = (
    proposal = Lux.initialstates(rng, m.proposal),
    prior    = Lux.initialstates(rng, m.prior),
    decoder  = Lux.initialstates(rng, m.decoder),
)

# like forward pass
function Lux.apply(m::VAEAC, (x, mask, ε), ps, st)
    x_masked = x .* mask
    xb = vcat(x_masked, mask)

    prop_out, st_prop = Lux.apply(m.proposal, xb, ps.proposal, st.proposal)
    μq = @view prop_out[1:m.ldim, :]
    logσq = @view prop_out[m.ldim+1:end, :]

    # ε = randn(Float32, size(μq))
    z = μq .+ exp.(logσq) .* ε

    prior_out, st_prior = Lux.apply(m.prior, mask, ps.prior, st.prior)
    μp = @view prior_out[1:m.ldim, :]
    logσp = @view prior_out[m.ldim+1:end, :]
    dec_in = vcat(z, mask)

    logits, st_dec = Lux.apply(m.decoder, dec_in, ps.decoder, st.decoder)
    return (logits, μq, logσq, μp, logσp), (proposal=st_prop, prior=st_prior, decoder=st_dec)
end

generate_mask(sz::Tuple{Int,Int}) = rand(Float32, sz) .> rand()
generate_mask(sz::Tuple{Int,Int,Int}) = rand(Float32, sz) .> rand(1,1,size(sz,3))

function bce_with_logits_masked(logits, x, mask)
    w = 1f0 .- mask
    per_elem = softplus.(logits) .- x .* logits
    s = sum(w .* per_elem)
    return s / size(x, 2)
end

function kl_diag_gaussians(μq, logσq, μp, logσp)
    σq2 = exp.(2f0 .* logσq)
    σp2 = exp.(2f0 .* logσp)
    t = (σq2 .+ (μq .- μp).^2) ./ σp2 .- 1f0 .+ 2f0 .* (logσp .- logσq)
    0.5f0 * sum(t) / size(μq, 2)
end

function loss_fn(model, ps, st, (x, mask, ε))
    (logits, μq, logσq, μp, logσp), st2 = Lux.apply(model, (x, mask, ε), ps, st)
    recon = bce_with_logits_masked(logits, x, mask)
    kl = kl_diag_gaussians(μq, logσq, μp, logσp)
    (recon + kl), st2, (; recon, kl)
end

function load_binary_mnist_matrix()
    ds = MNIST(; split=:train)
    imgs = Float32.(reshape(ds.features, :, size(ds.features, 3)))
    Float32.(imgs .> 0.5f0)
end

make_loader(x; batchsize=128, shuffle=true) = DataLoader(x; batchsize, shuffle)

function train_vaeac(; epochs=20, lr=0.001f0, batch_size=100)
    model = VAEAC(input_dim, latent_dim, hidden_dim)
    ps, st = Lux.setup(Random.default_rng(), model)

    Reactant.set_default_backend("gpu")
    dev = reactant_device()

    ps = ps |> dev
    st = st |> dev

    data = load_binary_mnist_matrix() #|> dev
    loader = make_loader(data; batchsize=batch_size, shuffle=true)
    loader_dev = DeviceIterator(dev, loader)

    opt = Optimisers.Descent(lr)  
    # opt = Optimisers.Adam(lr)
    ts = Lux.Training.TrainState(model, ps, st, opt)

    for epoch in 1:epochs
        tot = 0f0
        nb = 0
        for xb in loader_dev
            mask = Float32.(generate_mask(size(xb))) |> dev
            ε = randn(Float32, latent_dim, size(xb, 2)) |> dev

            # gs, loss, stats, ts = Lux.Training.compute_gradients(Lux.AutoEnzyme(), loss_fn, (xb, mask, ε), ts)
            # ts = Lux.Training.apply_gradients(ts, gs)
            
            _, loss, _, ts = Lux.Training.single_train_step!(Lux.AutoEnzyme(), loss_fn, (xb, mask, ε), ts)
            
            # tot += loss
            tot += Float32(loss)
            nb += 1
        end
        @info "epoch=$epoch avg_loss=$(tot/nb)"
    end
    return ts
end

#! fix batch size
ts = train_vaeac(epochs=3, lr=0.001f0, batch_size=100)















""" ================ Imputation and Sampling ================= """

function impute(ts::Lux.Training.TrainState, x, mask)
    (logits, μq, logσq, μp, logσp), _ = Lux.apply(ts.model, (x, mask), ts.parameters, ts.states)
    σ.(logits)
end

function sample_and_save_png(ts::Lux.Training.TrainState, x, mask; binary=true, path="vaeac_imputation.png")
    xhat = impute(ts, x, mask)
    g = reshape(xhat[:, 1], 28, 28)
    m = reshape(mask[:, 1], 28, 28)
    if binary
        g .= ifelse.(g .> 0.5f0, 1f0, 0f0)
    end
    img = Array{RGBf}(undef, 28, 28)

    for i in 1:28, j in 1:28
        if m[i, j] == 1
            img[i, j] = RGBf(0.5, 0, 0)
            if x[(j-1)*28 + i, 1] == 1
                img[i, j] = RGBf(1, 0, 0)
            end
        else
            gray_val = clamp(g[i, j], 0, 1)
            img[i, j] = RGBf(gray_val, gray_val, gray_val)
        end
    end
    reverse(Base.rotr90(img), dims=2)
end

function block_mask(dims=(28,28); top=12, left=10, h=10, w=10)
    m = trues(dims...)
    m[top:top+h-1, left:left+w-1] .= false
    Float32.(vec(m))  # 1 where missing
end

random_mask(n; D=784, rng=Random.default_rng()) = (m = falses(D); m[view(randperm(rng, D), 1:n)] .= true; m )

function sample_and_save(x, mask, ts; binary=true)

    (logits, μq, logσq, μp, logσp), _ = Lux.apply(ts.model, (x, mask), ts.parameters, ts.states)
    x_hat = σ.(logits)

    grayscale_image = x_hat[:, 1] isa AbstractVector ? reshape(x_hat[:, 1], 28, 28) : x_hat
    mask_image = mask[:, 1] isa AbstractVector ? reshape(mask[:, 1], 28, 28) : mask

    if binary
        grayscale_image .= ifelse.(grayscale_image .> 0.5, 1f0, 0f0)
    end

    color_image = Array{RGBf}(undef, 28, 28)

    for i in 1:28, j in 1:28
        if mask_image[i, j] == 1
            color_image[i, j] = RGBf(0.5, 0, 0)
            if x[(j-1)*28 + i, 1] == 1
                color_image[i, j] = RGBf(1, 0, 0)
            end
        else
            gray_val = clamp(grayscale_image[i, j], 0, 1)
            color_image[i, j] = RGBf(gray_val, gray_val, gray_val)
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