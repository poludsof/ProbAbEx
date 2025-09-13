
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

function Lux.apply(m::VAEAC, (x, mask, ε), ps, st)
    x_masked = x .* mask
    xb = vcat(x_masked, mask)

    prop_out, st_prop = Lux.apply(m.proposal, xb, ps.proposal, st.proposal)
    μq = @view prop_out[1:m.ldim, :]
    logσq = clamp.(prop_out[m.ldim+1:end, :], -10f0, 10f0 )

    z = μq .+ exp.(logσq) .* ε

    prior_out, st_prior = Lux.apply(m.prior, mask, ps.prior, st.prior)
    μp = @view prior_out[1:m.ldim, :]
    logσp = clamp.(prior_out[m.ldim+1:end, :], -10f0, 10f0 )
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

    data = load_binary_mnist_matrix()
    loader = make_loader(data; batchsize=batch_size, shuffle=true)
    loader_dev = DeviceIterator(dev, loader)

    # opt = Optimisers.Adam(lr)
    opt = Optimisers.OptimiserChain(
          Optimisers.WeightDecay(0.001f0),
          Optimisers.Adam(lr),
    )
    ts = Lux.Training.TrainState(model, ps, st, opt)

    for epoch in 1:epochs
        tot = 0f0
        nb = 0
        for xb in loader_dev
            mask = Float32.(generate_mask(size(xb))) |> dev
            ε = randn(Float32, latent_dim, size(xb, 2)) |> dev

            _, loss, _, ts = Lux.Training.single_train_step!(Lux.AutoEnzyme(), loss_fn, (xb, mask, ε), ts)
            tot += loss
            nb += 1
        end
        @info "epoch=$epoch avg_loss=$(tot/nb)"
    end
    return ts
end

# ts = train_vaeac(epochs=10, lr=0.001f0, batch_size=100)