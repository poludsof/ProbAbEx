
import ProbAbEx as PAE
import Lux: initialparameters, initialstates
# todo: GPU, Reactant.jl
# todo: IWAE

# ========== Model Definition ==========
struct VAEAC{P, Q, D} <: Lux.AbstractLuxLayer
    proposal_net::P
    prior_net::Q
    decoder_net::D
end
# struct VAEAC
#     proposal_net::Lux.Chain
#     prior_net::Lux.Chain
#     decoder_net::Lux.Chain
# end
# struct VAEAC
#     proposal_net::Lux.AbstractLuxLayer
#     prior_net::Lux.AbstractLuxLayer
#     decoder_net::Lux.AbstractLuxLayer
# end

# ========== Load & binarize MNIST ==========
function load_binary_mnist()
    train_x, _ = MNIST(split=:train)[:]
    train_x = Float32.(reshape(train_x, :, size(train_x, 3)) .> 0.5)  # binarize
    return train_x
end

function Lux.initialparameters(rng::AbstractRNG, m::VAEAC)
    println("CALLLING")
    (;
        proposal_net = Lux.initialparameters(rng, m.proposal_net),
        prior_net    = Lux.initialparameters(rng, m.prior_net),
        decoder_net  = Lux.initialparameters(rng, m.decoder_net),
    )
end

function Lux.initialstates(rng::AbstractRNG, m::VAEAC)
    (;
        proposal_net = Lux.initialstates(rng, m.proposal_net),
        prior_net    = Lux.initialstates(rng, m.prior_net),
        decoder_net  = Lux.initialstates(rng, m.decoder_net),
    )
end

function build_vaeac(input_dim::Int, latent_dim::Int, hidden_dim::Int; rng=Random.default_rng())
    proposal_net = Chain(
        Dense(2 * input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => 2 * latent_dim)
    )
    prior_net = Chain(
        Dense(2 * input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => 2 * latent_dim)
    )
    decoder_net = Chain(
        Dense(latent_dim + 2 * input_dim => hidden_dim, relu),
        Dense(hidden_dim => hidden_dim, relu),
        Dense(hidden_dim => input_dim)
    )
    model = VAEAC(proposal_net, prior_net, decoder_net)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

_split_gaussian(out, latent_dim) = begin
    μ  = @view out[1:latent_dim, :]
    ρ  = @view out[latent_dim+1:2*latent_dim, :]          # unconstrained
    σ  = softplus.(ρ) .+ 1f-5                              # strictly > 0
    logσ = log.(σ)
    # (optional extra guard)
    logσ = clamp.(logσ, -10f0, 2f0)
    (μ, logσ)
end

function _proposal_params(m::VAEAC, x::AbstractMatrix, b::AbstractMatrix, ps, st)
    xb = vcat(x, b)
    out, stp = Lux.apply(m.proposal_net, xb, ps.proposal_net, st.proposal_net)
    L = size(out, 1) ÷ 2
    μ, logσ = _split_gaussian(out, L)
    return μ, logσ, merge(st, (; proposal_net = stp))
end

function _prior_params(m::VAEAC, x::AbstractMatrix, b::AbstractMatrix, ps, st)
    xb = vcat(x .* (1 .- b), b)
    out, stp = Lux.apply(m.prior_net, xb, ps.prior_net, st.prior_net)
    L = size(out, 1) ÷ 2
    μ, logσ = _split_gaussian(out, L)
    return μ, logσ, merge(st, (; prior_net = stp))
end

function _decoder_logits(m::VAEAC, z::AbstractMatrix, x::AbstractMatrix, b::AbstractMatrix, ps, st)
    xb = vcat(x .* (1 .- b), b)
    dec_in = vcat(z, xb)
    logits, std = Lux.apply(m.decoder_net, dec_in, ps.decoder_net, st.decoder_net)
    return logits, merge(st, (; decoder_net = std))
end


function _kl_diag_gaussians(μq, logσq, μp, logσp)
    σq2 = exp.(2f0 .* logσq)
    σp2 = exp.(2f0 .* logσp)
    v = (σq2 .+ (μq .- μp).^2) ./ σp2 .+ 2f0 .* (logσp .- logσq) .- 1f0
    return 0.5f0 * mean(sum(v; dims=1))
end

function _bce_with_logits_masked(logits, targets, mask)
    # numerically-stable BCE-with-logits
    l = relu.(logits) .- logits .* targets .+ log.(1 .+ exp.(-abs.(logits)))
    cnt = Float32(sum(mask))
    return cnt == 0f0 ? 0f0 : sum(l .* mask) / cnt
end

function vaeac_loss(m::VAEAC, ps, st, x::AbstractMatrix, b::AbstractMatrix; rng=Random.default_rng())
    μq, logσq, st = _proposal_params(m, x, b, ps, st)
    μp, logσp, st = _prior_params(m, x, b, ps, st)
    ε = randn(rng, eltype(x), size(μq))
    z = μq .+ exp.(logσq) .* ε
    logits, st = _decoder_logits(m, z, x, b, ps, st)
    recon = _bce_with_logits_masked(logits, x, b)
    kl = _kl_diag_gaussians(μq, logσq, μp, logσp)
    return recon + kl, (recon=recon, kl=kl), st
end


function vaeac_sample(m::VAEAC, ps, st, x::AbstractMatrix, b::AbstractMatrix; rng=Random.default_rng(), n_samples::Int=1, sample::Bool=true)
    D, B = size(x)
    μp, logσp, st = _prior_params(m, x, b, ps, st)
    σp = exp.(logσp)
    outs = Vector{Matrix{eltype(x)}}(undef, n_samples)
    for s in 1:n_samples
        ε = randn(rng, eltype(x), size(μp))
        z = μp .+ σp .* ε
        logits, st = _decoder_logits(m, z, x, b, ps, st)
        p = sigmoid.(logits)
        xb_imp = if sample
            Float32.(rand(rng, size(p)) .< p)
            else
            Array{eltype(x)}(p)
        end
        outs[s] = x .* (1 .- b) .+ xb_imp .* b
    end
    return outs, st
end


function make_minibatches(x::AbstractMatrix, b::AbstractMatrix, batch::Int; shuffle::Bool=true, rng=Random.default_rng())
    N = size(x, 2)
    idx = collect(1:N)
    if shuffle
        Random.shuffle!(rng, idx)
    end
    Bs = [idx[s:min(s+batch-1, N)] for s in 1:batch:N]
    return ((@view(x[:, I]), @view(b[:, I])) for I in Bs)
end

_allfinite(x::Number) = isfinite(x)
_allfinite(x::AbstractArray) = all(isfinite, x)
_allfinite(x::NamedTuple) = all(_allfinite, values(x))
_allfinite(x::Tuple) = all(_allfinite, x)
_allfinite(::Any) = true


function train!(m::VAEAC, ps, st, x::AbstractMatrix, b::AbstractMatrix; epochs::Int=10, batch::Int=128, lr::Float64=1e-4, rng=Random.default_rng(), cb=nothing)
    opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.Adam(lr))
    opt_state = Optimisers.setup(opt, ps)

    for e in 1:epochs
        for (xb, bb) in make_minibatches(x, b, batch; shuffle=true, rng=rng)

            lossfun = p -> begin
                l, _, _ = vaeac_loss(m, p, st, xb, bb; rng=rng)
                l
            end

            gs = Zygote.gradient(lossfun, ps)[1]

            # skip updates if anything is non-finite
            if !_allfinite(gs)
                @warn "Non-finite grads; skipping step"
                continue
            end

            opt_state, ps = Optimisers.update!(opt_state, ps, gs)
        end

        if cb !== nothing
            l, parts, _ = vaeac_loss(m, ps, st, x, b; rng=rng)
            cb(e, l, parts)
        end

    end
    return ps, st
end

make_objective(rng=Random.default_rng()) =
    (m, p, s, (x, b)) -> begin
        loss, stats, s2 = vaeac_loss(m, p, s, x, b; rng=rng)  # you already return (loss, parts, st)
        return loss, s2, stats
    end

function train_ts!(m::VAEAC, ps, st, x::AbstractMatrix, b::AbstractMatrix;
                   epochs::Int=10, batch::Int=128, lr::Float64=1e-4,
                   rng=Random.default_rng(), cb=nothing)

    opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.Adam(lr))
    ts  = Lux.Training.TrainState(m, ps, st, opt)  # holds model/params/state/opt
    obj = make_objective(rng)

    for e in 1:epochs
        for (xb, bb) in make_minibatches(x, b, batch; shuffle=true, rng=rng)
            # compute grads wrt ts.parameters using your objective
            grads, loss, stats, ts = Lux.Training.compute_gradients(ADTypes.AutoZygote(), obj, (xb, bb), ts)

            # optional guard (your _allfinite handles NamedTuples/arrays)
            if !_allfinite(grads)
                @warn "Non-finite grads; skipping step"
                continue
            end

            # apply optimizer step to params (and advance ts.step)
            ts = Lux.Training.apply_gradients(ts, grads)
        end

        if cb !== nothing
            # report on full dataset (or a held-out batch)
            l, parts, _ = vaeac_loss(m, ts.parameters, ts.states, x, b; rng=rng)
            cb(e, l, parts)
        end
    end

    return ts.parameters, ts.states
end

D, L, H = 28*28, 64, 512          # input dim, latent dim, hidden dim
model, ps, st = build_vaeac(D, L, H)

@assert ps isa NamedTuple
@assert st isa NamedTuple
@show propertynames(ps)           # (:proposal_net, :prior_net, :decoder_net)
@show propertynames(st)       

B = 60_000                        # number of samples in dataset
x_mnist = load_binary_mnist()
b = Float32.(rand(D, B) .> 0.5)   # b :: (D, B) → the mask telling the model which entries are “missing”.


ps, st = train!(model, ps, st, x_mnist, b; epochs=5, batch=256, lr=1e-3, rng=Random.default_rng(),
    cb = (e, l, parts) -> @info("epoch=$e loss=$(round(l, digits=4)) recon=$(round(parts.recon, digits=4)) kl=$(round(parts.kl, digits=4))"))

ps, st = train_ts!(model, ps, st, x_mnist, b; epochs=5, batch=256, lr=1e-4,
    rng=Random.default_rng(),
    cb = (e, l, parts) -> @info "epoch=$e loss=$(round(l, digits=4)) recon=$(round(parts.recon, digits=4)) kl=$(round(parts.kl, digits=4))")

# --- Sampling convenience wrappers ---


function sample_missing(m::VAEAC, ps, st, x::AbstractVecOrMat, b::AbstractVecOrMat;
    n_samples::Int=1, rng=Random.default_rng(), stochastic::Bool=true, return_mean::Bool=false)
    X = x isa AbstractVector ? reshape(x, :, 1) : x
    B = b isa AbstractVector ? reshape(b, :, 1) : b
    outs, st = vaeac_sample(m, ps, st, X, B; rng=rng, n_samples=n_samples, sample=stochastic)
    if return_mean
        Y = reduce(+, outs) ./ n_samples
        return (x isa AbstractVector ? vec(Y) : Y), st
    else
        if n_samples == 1
            Y = outs[1]
            return (x isa AbstractVector ? vec(Y) : Y), st
        else
            return outs, st
        end
    end
end

function sample_missing!(dest::AbstractVecOrMat, m::VAEAC, ps, st, x::AbstractVecOrMat, b::AbstractVecOrMat;
    n_samples::Int=1, rng=Random.default_rng(), stochastic::Bool=true, return_mean::Bool=false)
    y, st = sample_missing(m, ps, st, x, b; n_samples=n_samples, rng=rng, stochastic=stochastic, return_mean=return_mean)
    dest .= y
    return dest, st
end

function block_mask(dims=(28,28); top=8, left=8, h=14, w=14)
    m = falses(dims...)
    m[top:top+h-1, left:left+w-1] .= true
    Float32.(vec(m))  # 1 where missing
end

b_vec = block_mask()
y_sample, st = sample_missing(model, ps, st, x₁, b_vec; n_samples=1, stochastic=true)
y_sample_bin = Float32.(y_sample .>= 0.5f0)
x₁ = x_mnist[:, 1]
b₁ = Float32.(rand(Bool, D))
y, st = sample_missing(model, ps, st, x₁, b₁; n_samples=10, stochastic=true, return_mean=true)

function show_imputation(m::VAEAC, ps, st, x::AbstractVector{<:Real}, b::AbstractVector{<:Real};
    n_samples::Int=20, stochastic::Bool=true, return_mean::Bool=true,
    binarize::Bool=true, savepath::Union{Nothing,String}=nothing,
    title::String="VAEAC Imputation")
    D = length(x)
    @assert D == length(b)
    side = round(Int, sqrt(D))
    @assert side*side == D "This viz assumes square images (e.g., 28x28)."

    x01 = clamp.(Float32.(x), 0f0, 1f0)
    b01 = Float32.(b)

    y, st = sample_missing(m, ps, st, x01, b01; n_samples=n_samples, stochastic=stochastic, return_mean=return_mean)
    y01 = Float32.(return_mean ? y : y[1])
    ybin = binarize ? Float32.(y01 .>= 0.5f0) : y01

    orig = reshape(x01, side, side)
    maskv = reshape(b01, side, side)
    masked = reshape(x01 .* (1f0 .- b01) .+ 0.5f0 .* b01, side, side)
    imputed= reshape(y01, side, side)
    impbin = reshape(ybin, side, side)

    fig = Figure(resolution=(900, 900), fontsize=14)
    fig[0,1] = Label(fig, title, tellwidth=false)

    ax1 = Axis(fig[1,1], title="Original"); image!(ax1, orig); hidespines!(ax1); hidedecorations!(ax1, grid=false)
    ax2 = Axis(fig[1,2], title="Mask (white=missing)"); image!(ax2, maskv); hidespines!(ax2); hidedecorations!(ax2, grid=false)
    ax3 = Axis(fig[2,1], title="Masked input"); image!(ax3, masked); hidespines!(ax3); hidedecorations!(ax3, grid=false)
    ax4 = Axis(fig[2,2], title=binarize ? "Imputed (binary)" : "Imputed"); image!(ax4, binarize ? impbin : imputed); hidespines!(ax4); hidedecorations!(ax4, grid=false)

    if savepath !== nothing
        save(savepath, fig)
    end
    display(fig)
    return (imputed=y01, imputed_binary=ybin, state=st)
end

res = show_imputation(model, ps, st, x₁, b_vec;
                      n_samples=20, stochastic=true, return_mean=true,
                      binarize=true, savepath="imputation_k1.png",
                      title="VAEAC Imputation")


sample_and_save(y_sample_bin, b₁; binary=true)

function sample_and_save(x_hat, mask; binary=true)

    # logits, _, _, _, _ = forward(x, mask, model)
    # x_hat = σ.(logits)

    grayscale_image = reshape(x_hat[:, 1], 28, 28)
    mask_image = reshape(mask[:, 1], 28, 28)

    if binary
        grayscale_image .= ifelse.(grayscale_image .> 0.5, 1f0, 0f0)
    end

    color_image = Array{RGBf}(undef, 28, 28)

    for i in 1:28, j in 1:28
        if mask_image[i, j]
            # color_image[i, j] = RGBf(1, 0, 0)
            if x[(j-1)*28 + i, 1] == 1
                color_image[i, j] = RGBf(1, 0, 0)
            else
                color_image[i, j] = RGBf(0.5, 0, 0)
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
