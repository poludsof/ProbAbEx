module ReactantExt

using ProbAbEx
using Reactant
using ProbAbEx.StaticBitSets
using NNlib: softmax

const _L_CACHE = Dict{Int, Any}()
const _SAMPLE_KERNEL = Dict{Tuple{Int,Int,Int}, Any}()

function _get_L(K::Int)
    L = get(_L_CACHE, K, nothing)
    if L === nothing
        L_cpu = tril(ones(Float32, K, K))
        L = ConcretePJRTArray(L_cpu)
        _L_CACHE[K] = L
    end
    L
end

function condition_logits_gpu(log_p, x_s, mask)
    x01 = vcat((x_s .<= 0)', (x_s .> 0)')
    vec(sum(transpose(mask) .* x01 .* log_p; dims=(1,2)))
end

function sample_all_from_logits(p, logits, mask, x_s, L, n::Int)
    probs = softmax(logits)
    cs = L * reshape(probs, :, 1)

    u_mix = rand(Float32, 1, n)
    idx = vec(sum(cs .< u_mix; dims=1) .+ 1)

    p_sel = p[:, idx]
    u = rand(Float32, size(p_sel))
    raw = 2f0 .* (u .< p_sel) .- 1f0

    ifelse.(mask, Float32.(x_s), raw)
end

function _get_sample_kernel(D::Int, K::Int, n::Int)
    key = (D, K, n)
    f = get(_SAMPLE_KERNEL, key, nothing)
    if f === nothing
        p0 = ConcretePJRTArray(zeros(Float32, D, K))
        logits0 = ConcretePJRTArray(zeros(Float32, K))
        mask0 = ConcretePJRTArray(fill(false, D))
        x0 = ConcretePJRTArray(zeros(Float32, D))
        L0 = _get_L(K)
        f = Reactant.@compile sample_all_from_logits(p0, logits0, mask0, x0, L0, n)
        _SAMPLE_KERNEL[key] = f
    end
    f
end

function ProbAbEx.condition(r::ProbAbEx.BernoulliMixture, x_s::ConcretePJRTArray, known_ii::SBitSet)
    D = length(x_s)
    mask_cpu = fill(false, D)
    for i in known_ii
        mask_cpu[i] = true
    end
    mask = ConcretePJRTArray(mask_cpu)
    logits = @jit condition_logits_gpu(r.log_p, x_s, mask)
    ProbAbEx.ConditionedBernoulliMixture(r, x_s, mask, logits)
end

function ProbAbEx.sample_all(r::ProbAbEx.ConditionedBernoulliMixture, n::Integer)
    D = size(r.r.p, 1)
    K = size(r.r.p, 2)
    L = _get_L(K)
    f = _get_sample_kernel(D, K, Int(n))
    f(r.r.p, r.w, r.mask, r.xₛ, L, Int(n))
end

function ProbAbEx.sample_all!(u::ConcretePJRTArray, r::ProbAbEx.ConditionedBernoulliMixture)
    n = size(u, 2)
    x = ProbAbEx.sample_all(r, n)
    copyto!(u, x)
    u
end

end
