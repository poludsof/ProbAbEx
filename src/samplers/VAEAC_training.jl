
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

# function VAEAC(idim::Int, ldim::Int, h::Int)
#     proposal = Lux.Chain(
#         Lux.Dense(2idim => h, relu),
#         Lux.Dense(h => h, relu),
#         Lux.Dense(h => 2ldim)
#     )
#     prior = Lux.Chain(
#         Lux.Dense(idim => h, relu),
#         Lux.Dense(h => h, relu),
#         Lux.Dense(h => 2ldim)
#     )
#     decoder = Lux.Chain(
#         Lux.Dense(ldim + idim => h, relu),
#         Lux.Dense(h => h, relu),
#         Lux.Dense(h => idim)
#     )


# todo
# reshape_proposal(x) = reshape(x, 28, 28, 2, size(x, 2))
# reshape_prior(x)    = reshape(x, 28, 28, 1, size(x, 2))
# flatten_conv(x)     = reshape(x, :, size(x, 4))
flatten_tensor(x) = reshape(x, :, size(x, 4))

function VAEAC(idim::Int, ldim::Int, h::Int)
    
    # 1. PROPOSAL: Sees Full Image + Mask
    proposal = Lux.Chain(
        Lux.Conv((3,3), 2 => 32, relu, pad=1),
        Lux.BatchNorm(32),
        Lux.Conv((3,3), 32 => 32, relu, stride=2, pad=1),
        Lux.Conv((3,3), 32 => 64, relu, pad=1),
        Lux.BatchNorm(64),
        Lux.Conv((3,3), 64 => 64, relu, stride=2, pad=1),
        Lux.WrappedFunction(flatten_tensor),
        Lux.Dense(7*7*64 => h, relu),
        Lux.Dense(h => 2*ldim)
    )

    # 2. PRIOR: Sees Masked Image + Mask
    # Input: (28, 28, 2, B)
    prior = Lux.Chain(
        Lux.Conv((3,3), 2 => 32, relu, pad=1),
        Lux.BatchNorm(32),
        Lux.Conv((3,3), 32 => 32, relu, stride=2, pad=1),
        Lux.Conv((3,3), 32 => 64, relu, pad=1),
        Lux.BatchNorm(64),
        Lux.Conv((3,3), 64 => 64, relu, stride=2, pad=1),
        Lux.WrappedFunction(flatten_tensor),
        Lux.Dense(7*7*64 => h, relu),
        Lux.Dense(h => 2*ldim)
    )

    # 3. DECODER: Sees Latent + Masked Image (flat) + Mask (flat)
    # Input Size: ldim(20) + idim(784) + idim(784) = 1588
    dec_in_ch = 2 + ldim
    
decoder = Lux.Chain(
        Lux.Conv((3,3), dec_in_ch => 64, relu, pad=1),
        Lux.BatchNorm(64),
        Lux.Conv((3,3), 64 => 64, relu, pad=1),
        Lux.BatchNorm(64),
        Lux.Conv((3,3), 64 => 1, pad=1) 
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

# function Lux.apply(m::VAEAC, (x, mask, ε), ps, st)
#     x_masked = x .* mask
#     xb = vcat(x_masked, mask)

#     prop_out, st_prop = Lux.apply(m.proposal, xb, ps.proposal, st.proposal)
#     μq = @view prop_out[1:m.ldim, :]
#     logσq = clamp.(prop_out[m.ldim+1:end, :], -10f0, 10f0 )

#     z = μq .+ exp.(logσq) .* ε

#     prior_out, st_prior = Lux.apply(m.prior, mask, ps.prior, st.prior)
#     μp = @view prior_out[1:m.ldim, :]
#     logσp = clamp.(prior_out[m.ldim+1:end, :], -10f0, 10f0 )
#     dec_in = vcat(z, mask) # todo shape cat
#     cat(z, mask, dims = 3) #todo

#     logits, st_dec = Lux.apply(m.decoder, dec_in, ps.decoder, st.decoder)

#     return (logits, μq, logσq, μp, logσp), (proposal=st_prop, prior=st_prior, decoder=st_dec)
# end

function Lux.apply(m::VAEAC, (x_flat, mask_flat, ε), ps, st)
    B = size(x_flat, 2)
    x_img = reshape(x_flat, 28, 28, 1, B)       # Full Image
    mask_img = reshape(mask_flat, 28, 28, 1, B) # Mask
    x_masked_img = x_img .* mask_img            # Masked Image

    prop_in = cat(x_img, mask_img, dims=3) 
    prop_out, st_prop = Lux.apply(m.proposal, prop_in, ps.proposal, st.proposal)
    
    μq = @view prop_out[1:m.ldim, :]
    logσq = clamp.(prop_out[m.ldim+1:end, :], -10f0, 10f0)
    z = μq .+ exp.(logσq) .* ε

    prior_in = cat(x_masked_img, mask_img, dims=3)
    prior_out, st_prior = Lux.apply(m.prior, prior_in, ps.prior, st.prior)
    
    μp = @view prior_out[1:m.ldim, :]
    logσp = clamp.(prior_out[m.ldim+1:end, :], -10f0, 10f0)

    z_reshaped = reshape(z, 1, 1, m.ldim, B)
    z_tiled = repeat(z_reshaped, 28, 28, 1, 1)
    dec_in = cat(x_masked_img, mask_img, z_tiled, dims=3)
    logits_spatial, st_dec = Lux.apply(m.decoder, dec_in, ps.decoder, st.decoder)
    logits_flat = reshape(logits_spatial, :, B)
    return (logits_flat, μq, logσq, μp, logσp),
           (proposal=st_prop, prior=st_prior, decoder=st_dec)
end

generate_mask(sz::Tuple{Int,Int}) = rand(Float32, sz) .> rand()
generate_mask(sz::Tuple{Int,Int,Int}) = rand(Float32, sz) .> rand(1,1,size(sz,3))

function bce_with_logits_masked(logits, x, mask)
    w = 1f0 .- mask
    per_elem = softplus.(logits) .- x .* logits
    s = sum(w .* per_elem)
    # return s / size(x, 2)
    return s / (sum(w) + eps(Float32))
end

function kl_diag_gaussians(μq, logσq, μp, logσp)
    σq2 = exp.(2f0 .* logσq)
    σp2 = exp.(2f0 .* logσp)
    t = (σq2 .+ (μq .- μp).^2) ./ σp2 .- 1f0 .+ 2f0 .* (logσp .- logσq)
    0.5f0 * sum(t) / size(μq, 2)
end

function loss_fn(model, ps, st, (x, mask, ε), β)
    (logits, μq, logσq, μp, logσp), st2 = Lux.apply(model, (x, mask, ε), ps, st)
    recon_avg = bce_with_logits_masked(logits, x, mask)
    recon_scaled = recon_avg * 784f0
    kl = kl_diag_gaussians(μq, logσq, μp, logσp)
    total_loss = recon_scaled + β * kl
    # (recon + β * kl), st2, (; recon, kl, β)
    return total_loss, st2, (; recon=recon_scaled, kl, total_loss)
end

function load_binary_mnist_matrix()
    ds = MNIST(; split=:train)
    imgs = Float32.(reshape(ds.features, :, size(ds.features, 3)))
    Float32.(imgs .> 0.5f0)
end

make_loader(x; batchsize=128, shuffle=true) = DataLoader(x; batchsize, shuffle)

function train_vaeac(; epochs=20, lr=0.001f0, batch_size=100)
    println("Starting VAEAC Training...")
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
          Optimisers.ClipGrad(1.0f0),
          Optimisers.WeightDecay(0.001f0),
          Optimisers.Adam(lr),
    )
    ts = Lux.Training.TrainState(model, ps, st, opt)

    println("Training for $epochs epochs...")
    for epoch in 1:epochs
        β = min(1f0, epoch / 15f0)
        local_loss_fn = (m, ps, st, batch) -> loss_fn(m, ps, st, batch, β)
        # tot = 0f0
        tot_loss = 0f0
        tot_recon = 0f0
        tot_kl = 0f0
        nb = 0
        for xb in loader_dev
            mask = Float32.(generate_mask(size(xb))) |> dev
            ε = randn(Float32, latent_dim, size(xb, 2)) |> dev

            _, loss, stats, ts = Lux.Training.single_train_step!(Lux.AutoEnzyme(), local_loss_fn, (xb, mask, ε), ts)
            # tot += loss
            
            tot_loss += stats.total_loss
            tot_recon += stats.recon
            tot_kl += stats.kl
            nb += 1
        end
        avg_loss = round(tot_loss/nb, digits=4)
        avg_recon = round(tot_recon/nb, digits=4)
        avg_kl = round(tot_kl/nb, digits=4)
        @info "Epoch $epoch | Total: $avg_loss | Recon: $avg_recon | KL: $avg_kl | Beta: $β"
        # @info "epoch=$epoch avg_loss=$(tot/nb)"
    end
    return ts
end