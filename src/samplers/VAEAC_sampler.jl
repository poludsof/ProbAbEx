
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
    out, stp = m.proposal_net(xb, ps.proposal_net, st.proposal_net)
    L = size(out, 1) ÷ 2
    μ, logσ = _split_gaussian(out, L)
    return μ, logσ, merge(st, (; proposal_net = stp))
end


function _prior_params(m::VAEAC, x::AbstractMatrix, b::AbstractMatrix, ps, st)
    xb = vcat(x .* (1 .- b), b)
    out, stp = m.prior_net(xb, ps.prior_net, st.prior_net)
    L = size(out, 1) ÷ 2
    μ, logσ = _split_gaussian(out, L)
    return μ, logσ, merge(st, (; prior_net = stp))
end


function _decoder_logits(m::VAEAC, z::AbstractMatrix, x::AbstractMatrix, b::AbstractMatrix, ps, st)
    xb = vcat(x .* (1 .- b), b)
    dec_in = vcat(z, xb)
    logits, std = m.decoder_net(dec_in, ps.decoder_net, st.decoder_net)
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


function train!(m::VAEAC, ps, st, x::AbstractMatrix, b::AbstractMatrix;
                epochs::Int=10, batch::Int=128, lr::Float64=1e-4, rng=Random.default_rng(), cb=nothing)
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


D, L, H = 28*28, 64, 512
model, ps, st = build_vaeac(D, L, H)

@assert ps isa NamedTuple
@assert st isa NamedTuple
@show propertynames(ps)           # (:proposal_net, :prior_net, :decoder_net)
@show propertynames(st)       

ps.proposal_net

B = 10_000
x = rand(Float32, D, B)
b = Float32.(rand(D, B) .> 0.5)

ps, st = train!(model, ps, st, x, b; epochs=5, batch=256, lr=1e-3, rng=Random.default_rng(),
    cb = (e, l, parts) -> @info("epoch=$e loss=$(round(l, digits=4)) recon=$(round(parts.recon, digits=4)) kl=$(round(parts.kl, digits=4))"))