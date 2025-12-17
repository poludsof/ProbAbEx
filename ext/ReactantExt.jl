module ReactantExt

using ProbAbEx
using Reactant
using ProbAbEx.StaticBitSets
using ProbAbEx.StatsBase
using Flux: softmax
using ProbAbEx: ConditionedUniformDistribution, UniformDistribution,
               ConditionedBernoulliMixture, BernoulliMixture, condition

# Use the helper from ProbAbEx (you already use this elsewhere)
const rdev = ProbAbEx.reactant_device()

############################
#   Conditioned Uniform
############################
@inline function ProbAbEx.condition(r::UniformDistribution,
                                    xₛ::AbstractArray,
                                    known_ii::SBitSet)
    # Use the existing CPU implementation, then move probabilities to Reactant device
    cond = condition(r, Array(xₛ), known_ii)
    ConditionedUniformDistribution(rdev(cond.p))
end

@inline function ProbAbEx.sample_all(r::ConditionedUniformDistribution{<:AbstractVector},
                                     n::Integer)
    # r.p is already on the Reactant device (because of condition above)
    p = r.p
    # create random matrix on host, then move to same device
    x = rdev(rand(eltype(p), length(p), n))

    _f(xᵢ, pᵢ) = 2 * (xᵢ < pᵢ) - 1
    _f.(x, p)
end

############################
#   Conditioned BernoulliMixture
##########################

# condition:
function ProbAbEx.condition(r::ProbAbEx.BernoulliMixture, 
                            xₛ::ConcretePJRTArray,
                            known_ii::SBitSet )
    idim = length(xₛ)
    mask = fill(false, idim)
    for i in known_ii
        mask[i] = true
    end
    ProbAbEx.condition(r, xₛ, mask)
end

function compute_condition_probs(log_p, xₛ, mask)
    _xₛ = vcat((xₛ .≤ 0)', (xₛ .> 0)')
    mask_T = transpose(mask)
    logits = vec(sum(mask_T .* _xₛ .* log_p; dims = (1, 2)))
    return softmax(logits)
end

function ProbAbEx.condition(r::ProbAbEx.BernoulliMixture, 
                            xₛ::ConcretePJRTArray,
                            mask::Vector{Bool} )
    if !(mask isa ConcretePJRTArray)
        mask_gpu = ConcretePJRTArray(mask)
    else
        mask_gpu = mask
    end

    pzx_gpu = @jit compute_condition_probs(r.log_p, xₛ, mask_gpu)
    pzx_cpu = Array(pzx_gpu) 
    w = StatsBase.Weights(pzx_cpu)
    ProbAbEx.ConditionedBernoulliMixture(r, xₛ, mask_gpu, w)
end

# sample_all:
function compute_samples_batch(p, mask, x_cond, cids)
    T = eltype(p)
    p_selected = p[:, cids]
    unif = rand(T, size(p_selected))
    val_2 = T(2)
    val_1 = T(1)
    raw_samples = val_2 .* (unif .< p_selected) .- val_1
    x_cond_casted = T.(x_cond)
    return ifelse.(mask, x_cond_casted, raw_samples)
end
# function compute_samples_batch(p, mask, x_cond, cids)
#     p_selected = p[:, cids]
#     unif = rand(eltype(p), size(p_selected))
#     raw_samples = 2.0f0 .* (unif .< p_selected) .- 1.0f0
#     return ifelse.(mask, x_cond, raw_samples)
# end

function sample_all(::ProbAbEx.ConditionedBernoulliMixture{T, <:ConcretePJRTArray}, n::Integer) where T
    cids_cpu = StatsBase.sample(r.w, n)
    cids_gpu = ConcretePJRTArray(cids_cpu)
    samples_gpu = @jit compute_samples_batch(r.r.p, r.mask, r.xₛ, cids_gpu)
    return samples_gpu
end

function ProbAbEx.sample_all!(
    u::ConcretePJRTArray, 
    r::ProbAbEx.ConditionedBernoulliMixture{T, <:ConcretePJRTArray}
) where T
    n = size(u, 2)
    cids_cpu = StatsBase.sample(r.w, n)
    cids_gpu = ConcretePJRTArray(cids_cpu)
    
    new_samples = @jit compute_samples_batch(r.r.p, r.mask, r.xₛ, cids_gpu)
    copyto!(u, new_samples)
    
    return u
end

end