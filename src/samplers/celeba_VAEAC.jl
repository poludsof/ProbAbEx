
const Hc = 32
const Wc = 32
const Cc = 3
const Dc = Hc * Wc * Cc
const LDIM = 32
const HIDDEN = 400
const BATCH = 64
const EPOCHS = 20

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
    VAEAC(proposal, prior, decoder, idim, ldim)
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
    pred, st_dec = Lux.apply(m.decoder, dec_in, ps.decoder, st.decoder)
    (pred, μq, logσq, μp, logσp), (proposal=st_prop, prior=st_prior, decoder=st_dec)
end

function load_celeba_images(root::AbstractString; out_size=(32,32), limit::Union{Nothing,Int}=nothing, shuffle::Bool=true)
    
    files = filter(f -> endswith(lowercase(f), ".jpg") || endswith(lowercase(f), ".png"),
                   readdir(root; join=true))
                   
    shuffle && Random.shuffle!(files)

    limit !== nothing && (files = files[1:limit])

    H, W = out_size; C = 3; D = H*W*C
    X = Array{Float32}(undef, D, length(files))

    for (i, fp) in enumerate(files)
        img = load(fp)                  # could be H×W Colorant, or H×W×C numeric
        img = RGB.(img)                 # ensure 2-D Colorant array H×W

        # center crop to square on the *first two* dims
        H0, W0 = size(img, 1), size(img, 2)
        s = min(H0, W0)
        top  = (H0 - s) ÷ 2 + 1
        left = (W0 - s) ÷ 2 + 1
        cropped = @view img[top:top+s-1, left:left+s-1]   # <-- no third-dim index

        # resize the 2-D Colorant array
        img_res = ImageTransformations.imresize(cropped, (H, W))  # H×W Colorant

        # to H×W×C Float32 then flatten to D
        HWC = Float32.(permutedims(channelview(img_res), (2,3,1)))  # H×W×C, Float32 in [0,1]
        X[:, i] = reshape(HWC, D)
    end
    return X
end

data = load_celeba_images("/home/poludsof/ProbAbEx/src/datasets/img_align_celeba", out_size=(32, 32), limit=1000, shuffle=true)

function rand_rect_mask(H, W; min_area_frac=0.25)
    A = H * W
    while true
        y1 = rand(1:H); y2 = rand(1:H); x1 = rand(1:W); x2 = rand(1:W)
        ylo, yhi = minmax(y1, y2); xlo, xhi = minmax(x1, x2)
        if (yhi - ylo + 1) * (xhi - xlo + 1) >= floor(Int, min_area_frac * A)
            m2 = trues(H, W)
            m2[ylo:yhi, xlo:xhi] .= false
            return m2
        end
    end
end

function rect_mask_batch(H, W, C, B; min_area_frac=0.25)
    cols = [
        vec(Float32.(repeat(rand_rect_mask(H, W; min_area_frac=min_area_frac), 1, 1, C)))
        for _ in 1:B
    ]
    hcat(cols...)  # (H*W*C, B)
end

function mse_on_missing(pred, x, mask)
    @assert size(pred) == size(x)
    @assert size(mask) == size(x)
    B = size(x, 2)
    sum((1f0 .- mask) .* (pred .- x).^2) / B
end

function kld(μq, logσq, μp, logσp)
    σq2 = exp.(2f0 .* logσq)
    σp2 = exp.(2f0 .* logσp)
    t = (σq2 .+ (μq .- μp).^2) ./ σp2 .- 1f0 .+ 2f0 .* (logσp .- logσq)
    0.5f0 * sum(t) / size(μq, 2)
end

to_flat(x; H=Hc, W=Wc, C=Cc) = ndims(x) == 4 ? reshape(x, H*W*C, size(x, 4)) : x

function loss_fn_celeba(model, ps, st, x)
    x = to_flat(x)
    B = size(x, 2)
    mask = rect_mask_batch(Hc, Wc, Cc, B)
    (pred, μq, logσq, μp, logσp), st2 = Lux.apply(model, (x, mask), ps, st)
    recon = mse_on_missing(pred, x, mask)
    kl = kld(μq, logσq, μp, logσp)
    (recon + kl), st2, (; recon, kl)
end

function train_celeba(root::AbstractString; out_size=(Hc, Wc), limit=nothing, epochs=EPOCHS, batchsize=BATCH, ldim=LDIM, h=HIDDEN, lr=LR)
    # X = load_celeba_images(root; out_size=out_size, limit=limit, shuffle=true)
    data = load_celeba_images("/home/poludsof/ProbAbEx/src/datasets/img_align_celeba", out_size=(32, 32), limit=1000, shuffle=true)
    X = data
    println("Loaded data size: ", size(data))
    model = VAEAC(size(X, 1), ldim, h)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model)
    ts = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(lr))
    idx = collect(1:size(X, 2))
    for e in 1:epochs
        shuffle!(idx)
        tot = 0f0; nb = 0
        for k in 1:batchsize:length(idx)
            bi = idx[k:min(k + batchsize - 1, end)]
            xb = X[:, bi]
            gs, loss, stats, ts = Lux.Training.compute_gradients(Lux.Training.AutoZygote(), loss_fn_celeba, xb, ts)
            ts = Lux.Training.apply_gradients(ts, gs)
            tot += loss; nb += 1
        end
        @info "epoch=$(e) loss=$(tot/nb)"
    end
    ts
end


# pathway = "/home/poludsof/ProbAbEx/src/datasets/img_align_celeba"
# ts = train_celeba(pathway; out_size=(32, 32), limit=100, epochs=10, lr = 0.001f0)

function impute_celeba(ts::Lux.Training.TrainState, x, mask)
    x = to_flat(x)
    mask = to_flat(mask)
    (y, μq, logσq, μp, logσp), _ = Lux.apply(ts.model, (x, mask), ts.parameters, ts.states)
    y
end

function sample_celeba(ts::Lux.Training.TrainState, x; min_area_frac=0.25)
    x = to_flat(x)
    B = size(x, 2)
    mask = rect_mask_batch(Hc, Wc, Cc, B; min_area_frac=min_area_frac)
    y = impute_celeba(ts, x, mask)
    y, mask
end

function to_image(vecx::AbstractVector{<:Real}; H=Hc, W=Wc, C=Cc)
    a = clamp.(reshape(vecx, H, W, C), 0f0, 1f0)
    colorview(RGB, permutedims(a, (3, 1, 2)))
end
