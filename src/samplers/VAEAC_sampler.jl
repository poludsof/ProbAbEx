
struct VAEACSampler{M,P,S}
    model::M
    ps::P
    st::S
end

function VAEACSampler(train_state)
    model, ps, st = train_state.model, train_state.ps, train_state.st
    VAEACSampler(model, ps, st)
end

struct ConditionedVAEAC{R,X,M}
    r::R
    xₛ::X
    mask::M
end

function condition(r::VAEACSampler, xₛ, known_ii::SBitSet)
    idim = length(xₛ)
    mask = fill(false, idim)
    for i in known_ii
        mask[i] = true
    end
    ConditionedVAEAC(r, xₛ, mask)
end

function condition(r::VAEACSampler, xₛ, mask::Vector{Bool})
    ConditionedVAEAC(r, xₛ, mask)
end

"""
    sample_all(r::AbstractSampler, n::Integer)

    Sample `n` samples including fixed (condition) part, which is 
    copied
"""
function sample_all(r::ConditionedVAEAC, n::Integer)
    u = similar(r.xₛ, length(r.xₛ), n)
    sample_all!(u, r)
end

"""
    sample_all!(u, r::AbstractSampler)

    Sample `size(u,2)` samples including fixed (condition) part, which is 
    copied
"""
function sample_all!(u, r::ConditionedVAEAC)
    mk = r.mask
    n = size(u, 2)
    size(u, 1) == length(mk) || error("dimension of u does not match the dimension of the sampler")

    x  = repeat(reshape(Float32.(r.xₛ), length(r.xₛ), 1), 1, n)
    m  = repeat(reshape(Float32.(mk), length(r.xₛ), 1), 1, n)

    x_hat = impute!(r.r.model, r.r.ps, r.r.st, x, m)

    x_hat .= ifelse.(x_hat .> 0.5f0, 1, -1)
end

function impute!(model, ps, st, x::AbstractMatrix, m::AbstractMatrix)
    ldim = getfield(model, :ldim)
    ε = randn(Float32, ldim, size(x, 2))
    (logits, μq, logσq, μp, logσp), _ = Lux.apply(model, (x, m, ε), ps, st)
    return σ.(logits)
end