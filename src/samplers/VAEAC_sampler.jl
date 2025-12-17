
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
    u = zeros(Float32, length(r.xₛ), n)
    sample_all!(u, r)
end

"""
    sample_all!(u, r::AbstractSampler)

    Sample `size(u,2)` samples including fixed (condition) part, which is 
    copied
"""
function sample_all!(u, r::ConditionedVAEAC)
    mk = r.mask
    n  = size(u, 2)
    
    x_val_cpu = Array(r.xₛ) 
    m_val_cpu = Array(mk)

    x_batch_cpu = repeat(reshape(Float32.(x_val_cpu), :, 1), 1, n)
    m_batch_cpu = repeat(reshape(Float32.(m_val_cpu), :, 1), 1, n)

    dev = ProbAbEx.reactant_device()
    x_dev = dev(x_batch_cpu)
    m_dev = dev(m_batch_cpu)

    x_hat_dev = impute!(r.r.model, r.r.ps, r.r.st, x_dev, m_dev)

    x_hat_cpu = Array(x_hat_dev)
    u .= ifelse.(x_hat_cpu .> 0.5f0, 1f0, -1f0)
    
    return u
end

# function impute!(model, ps, st, x::AbstractMatrix, m::AbstractMatrix)
#     ldim = getfield(model, :ldim)
#     ε = randn(Float32, ldim, size(x, 2))
#     (logits, μq, logσq, μp, logσp), _ = Lux.apply(model, (x, m, ε), ps, st)
#     return σ.(logits)
# end

to01(x) = (x .+ 1f0) ./ 2f0 # from [-1,1] to [0,1]
from01(x) = 2f0 .* x .- 1f0 # from [0,1] to [-1,1]

# function impute!(model, ps, st, x::AbstractMatrix, m::AbstractMatrix)
#     dev = ProbAbEx.reactant_device()
#     x_dev = dev(x)    # safe even if x already on device (may be a no-op or copy)
#     m_dev = dev(m)
#     ldim = getfield(model, :ldim)
#     ε = dev(randn(Float32, ldim, size(x_dev, 2)))    # noise on same device
#     (logits, μq, logσq, μp, logσp), _ = Lux.apply(model, (x_dev, m_dev, ε), ps, st)
#     return σ.(logits)
# end

# helper to move nested params/state to device and materialize arrays
function move_to_device(x, dev)
    if x isa AbstractArray
        xa = dev(x)
        return copy(xa)
    elseif x isa NamedTuple
        kv = values(x)
        newvals = ntuple(i -> move_to_device(kv[i], dev), length(kv))
        return NamedTuple{keys(x)}(newvals)
    elseif x isa Tuple
        return tuple(move_to_device.(collect(x), Ref(dev))...)
    elseif x isa Dict
        d = Dict{Any,Any}()
        for (k,v) in x
            d[k] = move_to_device(v, dev)
        end
        return d
    else
        return dev(x)
    end
end

# robust CPU-side reshape helper: infers channels automatically
function reshape_proposal_cpu(x_cpu::AbstractMatrix; H::Int=28, W::Int=28)
    D, N = size(x_cpu)
    hw = H * W
    if D % hw != 0
        error("reshape_proposal_cpu: flattened dimension D=$D is not divisible by H*W=$hw")
    end
    C = div(D, hw)
    return reshape(x_cpu, H, W, C, N)
end

# device-safe impute! that handles both flattened (D,N) and shaped (H,W,C,N)
function impute!(model, ps, st, x::AbstractArray, m::AbstractArray; H::Int=28, W::Int=28)
    dev = ProbAbEx.reactant_device()

    # 1) Bring to CPU so reshape is performed on host (avoids device ReshapedArray copy iteration)
    x_cpu = Array(x)
    m_cpu = Array(m)

    # 2) If flattened (D,N) or D divisible by H*W, reshape on CPU to (H,W,C,N) and then transfer
    D, N_cpu = size(x_cpu)
    if D % (H * W) == 0
        x_img_cpu = reshape_proposal_cpu(x_cpu; H=H, W=W)
        m_img_cpu = reshape_proposal_cpu(m_cpu; H=H, W=W)
        x_use = dev(x_img_cpu)    # bulk transfer to device -> ConcretePJRTArray{Float32,4,1}
        m_use = dev(m_img_cpu)
    else
        # leave flattened, transfer as (D,N)
        x_use = dev(x_cpu)        # ConcretePJRTArray{Float32,2,1}
        m_use = dev(m_cpu)
    end

    # 3) move params/state to device
    ps_dev = move_to_device(ps, dev)
    st_dev = move_to_device(st, dev)

    # 4) infer batch size N robustly: for shaped arrays N is last dim, for 2D it's second dim
    nd = ndims(x_use)
    N = nd == 2 ? size(x_use, 2) : size(x_use, nd)

    # 5) noise on device with correct batch dimension
    ldim = getfield(model, :ldim)
    ε = dev(randn(Float32, ldim, N))
    ε = copy(ε)   # ensure concrete device buffer

    @info "impute! inputs types" typeof(x_use) typeof(m_use) typeof(ε)
    @info "impute! sizes" size(x_use) size(m_use) size(ε)

    (logits, μq, logσq, μp, logσp), _ = Lux.apply(model, (x_use, m_use, ε), ps_dev, st_dev)

    return σ.(logits)
end
function ShapleyHeuristic(sm, sampler, num_samples, verbose = false)
    ShapleyHeuristic(sm, sampler, num_samples, verbose)
end