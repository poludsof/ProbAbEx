
# todo: IWAE
struct TrainingLog
    epoch::Int
    β::Float32
    avg_loss::Float32
    avg_recon::Float32
    avg_kl::Float32
end

""" ========= Hyperparameters ========= """
const input_dim = 28 * 28
const latent_dim = 64
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
    # Input Size: ldim(64) + idim(784) + idim(784) = 1616
    dec_in_ch = 2 + ldim

    # decoder = Lux.Chain(
    #     Lux.Conv((3,3), dec_in_ch => 64, relu, pad=1),
    #     Lux.BatchNorm(64),
    #     Lux.Conv((3,3), 64 => 64, relu, pad=1),
    #     Lux.BatchNorm(64),
    #     Lux.Conv((3,3), 64 => 1, pad=1)
    # )
    decoder = Lux.Chain(
        Lux.Dense(ldim + 2 * idim => h, relu),
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

    # z_reshaped = reshape(z, 1, 1, m.ldim, B)
    # z_tiled = repeat(z_reshaped, 28, 28, 1, 1)
    # dec_in = cat(x_masked_img, mask_img, z_tiled, dims=3)
    # logits_spatial, st_dec = Lux.apply(m.decoder, dec_in, ps.decoder, st.decoder)
    # logits_flat = reshape(logits_spatial, :, B)
    dec_in = vcat(z, x_flat .* mask_flat, mask_flat)
    logits_flat, st_dec = Lux.apply(m.decoder, dec_in, ps.decoder, st.decoder)
    return (logits_flat, μq, logσq, μp, logσp),
           (proposal=st_prop, prior=st_prior, decoder=st_dec)
end

## BIG mask
# generate_mask(sz::Tuple{Int,Int}) = rand(Float32, sz) .> rand()
# generate_mask(sz::Tuple{Int,Int,Int}) = rand(Float32, sz) .> rand(1,1,size(sz,3))

## SMALL mask
# function generate_mask(sz::Tuple{Int,Int}; min_px=5, max_px=20)
#     n_pixels = rand(min_px:max_px)
#     mask = zeros(Float32, sz)
#     indices = randperm(sz[1])[1:n_pixels]
#     mask[indices, :] .= 1f0
#     mask
# end

# function generate_mask(sz::Tuple{Int,Int,Int}; min_px=5, max_px=20)
#     mask = zeros(Float32, sz)
#     for b in 1:sz[2]
#         n_pixels = rand(min_px:max_px)
#         indices = randperm(sz[1])[1:n_pixels]
#         mask[indices, b, :] .= 1f0
#     end
#     mask
# end

function generate_mask(sz::Tuple{Int,Int}; p_mixed=0.5)
    num_pixels, batch_size = sz
    mask = zeros(Float32, num_pixels, batch_size)

    for b in 1:batch_size
        if rand() < p_mixed
            keep_prob = rand(0.05:0.01:0.95)
            mask[:, b] .= Float32.(rand(num_pixels) .< keep_prob)
        else
            m2d = ones(Float32, 28, 28)

            h, w = rand(10:20), rand(10:20)
            y = rand(1:(28 - h + 1))
            x = rand(1:(28 - w + 1))

            m2d[y:y+h-1, x:x+w-1] .= 0f0
            mask[:, b] .= reshape(m2d, :)
        end
    end

    return mask
end

# function generate_mask(sz::Tuple{Int,Int,Int}; p_mixed=0.5)
#     # sz = (784, batch_size, 1)
#     mask = zeros(Float32, sz)
#     num_pixels, batch_size, _ = sz
    
#     for b in 1:batch_size
#         # Решаем, какой тип маски использовать для этого примера в батче
#         if rand() < p_mixed
#             # 1. СЛУЧАЙНЫЕ ПИКСЕЛИ (как было раньше, но с широким диапазоном)
#             # Оставляем от 5% до 95% изображения видимым
#             keep_prob = rand(0.05:0.01:0.95)
#             mask[:, b, 1] .= Float32.(rand(num_pixels) .< keep_prob)
#         else
#             # 2. СТРУКТУРНЫЕ БЛОКИ (Inpainting)
#             # Создаем маску 28x28, чтобы вырезать куски
#             m2d = ones(Float32, 28, 28)
            
#             # Выбираем случайный размер и координаты "дырки"
#             h, w = rand(10:20), rand(10:20)
#             y = rand(1:(28 - h + 1))
#             x = rand(1:(28 - w + 1))
            
#             # Затираем блок (0 означает "неизвестно" для вашей функции потерь)
#             m2d[y:y+h-1, x:x+w-1] .= 0f0
            
#             # Если ваша логика: 1 = "знаю", 0 = "не знаю" (как в вашем коде)
#             # Если наоборот, используйте m2d[y:y+h-1, x:x+w-1] .= 1f0 выше
#             mask[:, b, 1] .= reshape(m2d, :)
#         end
#     end
#     return mask
# end

# function bce_with_logits_masked(logits, x, mask)
#     w = 1f0 .- mask
#     per_elem = softplus.(logits) .- x .* logits
#     s = sum(w .* per_elem)
#     # return s / size(x, 2)
#     return s / (sum(w) + eps(Float32))
# end

function kl_diag_gaussians(μq, logσq, μp, logσp)
    σq2 = exp.(2f0 .* logσq)
    σp2 = exp.(2f0 .* logσp)
    t = (σq2 .+ (μq .- μp).^2) ./ σp2 .- 1f0 .+ 2f0 .* (logσp .- logσq)
    0.5f0 * sum(t) / size(μq, 2)
end

# function loss_fn(model, ps, st, (x, mask, ε), β)
#     (logits, μq, logσq, μp, logσp), st2 = Lux.apply(model, (x, mask, ε), ps, st)
#     recon_avg = bce_with_logits_masked(logits, x, mask)
#     recon_scaled = recon_avg * 784f0
#     kl = kl_diag_gaussians(μq, logσq, μp, logσp)
#     total_loss = recon_scaled + β * kl
#     # (recon + β * kl), st2, (; recon, kl, β)
#     return total_loss, st2, (; recon=recon_scaled, kl, total_loss)
# end

function bce_with_logits_masked(logits, x, mask)
    unknown = 1f0 .- mask
    pos_weight = 1.5f0

    per_elem = (1f0 .- x) .* softplus.(logits) .+
               x .* pos_weight .* softplus.(-logits)

    return sum(unknown .* per_elem) / size(x, 2)
end

function loss_fn(model, ps, st, (x, mask, ε), βv)
    β = sum(βv)
    (logits, μq, logσq, μp, logσp), st2 = Lux.apply(model, (x, mask, ε), ps, st)
    recon = bce_with_logits_masked(logits, x, mask)
    kl = kl_diag_gaussians(μq, logσq, μp, logσp)
    total_loss = recon + β * kl
    return total_loss, st2, (; recon, kl, total_loss)
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

    # lr_schedule(epoch) = lr * (1f0 + cos(π * epoch / epochs)) / 2f0

    # opt = Optimisers.Adam(lr)
    opt = Optimisers.OptimiserChain(
          Optimisers.ClipGrad(1.0f0),
        #   Optimisers.WeightDecay(0.0001f0),
          Optimisers.Adam(lr),
    )
    ts = Lux.Training.TrainState(model, ps, st, opt)

    logs = TrainingLog[]
    println("Training for $epochs epochs NEW...")
    for epoch in 1:epochs
        # β = min(1f0, epoch / 15f0)
        β = min(0.05f0, Float32(epoch) / 200f0)
        # local_loss_fn = (m, ps, st, batch) -> loss_fn(m, ps, st, batch, β)
        tot_loss = 0f0
        tot_recon = 0f0
        tot_kl = 0f0
        nb = 0
        
        for xb in loader_dev
            mask = Float32.(generate_mask(size(xb))) |> dev
            ε = randn(Float32, latent_dim, size(xb, 2)) |> dev
            βv = fill(Float32(β), 1) |> dev
            local_loss_fn = (m, ps, st, batch) -> loss_fn(m, ps, st, batch, βv)

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
        push!(logs, TrainingLog(epoch, β, avg_loss, avg_recon, avg_kl))
        @info "Epoch $epoch | Total: $avg_loss | Recon: $avg_recon | KL: $avg_kl | Beta: $β"

        # @info "epoch=$epoch avg_loss=$(tot/nb)"
    end

    return ts, logs
end

