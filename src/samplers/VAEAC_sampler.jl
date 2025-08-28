
import ProbAbEx as PAE
import Lux: initialparameters, initialstates
# todo: GPU, Reactant.jl
# todo: IWAE

# ========= Hyperparameters =========

const input_dim = 28 * 28
const latent_dim = 20
const hidden_dim = 400
const batch_size = 100
const epochs = 20
const learning_rate = 0.001f0

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

function Lux.apply(m::VAEAC, (x, mask), ps, st)
    x_masked = x .* mask
    xb = vcat(x_masked, mask)
    prop_out, st_prop = Lux.apply(m.proposal, xb, ps.proposal, st.proposal)
    μq = @view prop_out[1:m.ldim, :]
    logσq = @view prop_out[m.ldim+1:end, :]
    ε = randn(Float32, size(μq))
    z = μq .+ exp.(logσq) .* ε
    prior_out, st_prior = Lux.apply(m.prior, mask, ps.prior, st.prior)
    μp = @view prior_out[1:m.ldim, :]
    logσp = @view prior_out[m.ldim+1:end, :]
    dec_in = vcat(z, mask)
    logits, st_dec = Lux.apply(m.decoder, dec_in, ps.decoder, st.decoder)
    return (logits, μq, logσq, μp, logσp), (proposal=st_prop, prior=st_prior, decoder=st_dec)
end

generate_mask(sz::Tuple{Int,Int}) = Float32.(rand(Bool, sz))

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

function loss_fn(model, ps, st, x)
    mask = generate_mask(size(x))
    (logits, μq, logσq, μp, logσp), st2 = Lux.apply(model, (x, mask), ps, st)
    recon = bce_with_logits_masked(logits, x, mask)
    kl = kl_diag_gaussians(μq, logσq, μp, logσp)
    (recon + kl), st2, (; recon, kl)
end

function load_binary_mnist_matrix()
    ds = MNIST(; split=:train)
    imgs = Float32.(reshape(ds.features, :, size(ds.features, 3)))
    Float32.(imgs .> 0.5f0)
end

function make_loader(x; batchsize=batch_size, shuffle=true)
    DataLoader(x; batchsize=batchsize, shuffle=shuffle)
end

function train(; epochs=20, lr=learning_rate)
    model = VAEAC(input_dim, latent_dim, hidden_dim)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    ts = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(lr))
    data = load_binary_mnist_matrix()
    loader = make_loader(data; batchsize=batch_size, shuffle=true)
    for epoch in 1:epochs
        tot = 0f0
        nb = 0
        for xb in loader
            gs, loss, stats, ts = Lux.Training.compute_gradients(
                Lux.Training.AutoZygote(), loss_fn, xb, ts)
            ts = Lux.Training.apply_gradients(ts, gs)
            tot += loss
            nb += 1
        end
        @info "epoch=$epoch avg_loss=$(tot/nb)"
    end
    return ts
end


ts = train()


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
    img = Array{RGB{Float32}}(undef, 28, 28)
    @inbounds for i in 1:28, j in 1:28
        img[i, j] = m[i, j] > 0.5f0 ? RGB(1, 0, 0) : RGB(g[i, j], g[i, j], g[i, j])
    end
    img
end

x = load_binary_mnist_matrix()[:, 1:1]
mask = generate_mask(size(x))
path = sample_and_save_png(ts, x, mask; binary=true, path="impute.png")


xhat = impute(ts, x, mask)
show_image(xhat)





display_image(x_imp[:, 1])
show_image(p[:, 1])

function display_image(img)

    matrix_image = reshape(img[:, 1], 28, 28)

    # color_image = Array{RGBf}(undef, 28, 28)

    # for i in 1:28, j in 1:28
    #     gray_val = clamp(matrix_image[i, j], 0, 1)
    #     color_image[i, j] = RGBf(gray_val, gray_val, gray_val)
    # end

    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1], title = "Masked MNIST Reconstruction", yreversed = true, aspect = DataAspect())
    image!(ax, matrix_image, interpolate = false)
    hidespines!(ax); hidedecorations!(ax)

    fig
end

function show_image(x; dims=(28,28), title::String="Image", savepath::Union{Nothing,String}=nothing, order::Symbol=:col, clamp01::Bool=true)
    X = x isa AbstractVector ? reshape(Float32.(x), dims...) : (size(x) == dims ? Float32.(x) : reshape(Float32.(x), dims...))
    if order === :row
        X = permutedims(X, (2,1))
    end
    if clamp01
        X = clamp.(X, 0f0, 1f0)
    end
    
    fig = Figure(resolution=(320,320), fontsize=14)
    ax = Axis(fig[1,1], title=title)
    image!(ax, X); hidespines!(ax); hidedecorations!(ax, grid=false)
    
    if savepath !== nothing
        save(savepath, fig)
    end
    
    display(fig)
    return fig
end