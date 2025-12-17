using StatsBase

"""
struct BernoulliMixture{T<:Real} <: AbstractSampler
    log_p::Array{T, 3}
    p::Matrix{T}
end

Implements BernoulliMixture with uniform weights.
`log_p` holds logarithms (`logsoftmax`) of probabilities 
(the third dimension is probability of one and zero, i.e. `size(log_p,3)=2`),
which simplifies the conditioning, which is crucial operation.

The conditioning is
"""
# struct BernoulliMixture{T<:Real, LP<:AbstractArray{T,3}, P<:AbstractMatrix{T}}
#     log_p::LP
#     p::P
# end

# function BernoulliMixture(centers)
#     log_p = logsoftmax(centers, dims = 1)
#     p = softmax(centers, dims = 1)[2,:,:]
#     BernoulliMixture(log_p, p)
# end

struct BernoulliMixture{T<:Real, LP<:AbstractArray{T}, P<:AbstractArray{T}}
    log_p::LP
    p::P
end

function compute_mixture_params(centers)
    log_p = logsoftmax(centers; dims = 1)
    p = softmax(centers; dims = 1)[2, :, :]
    return log_p, p
end

function BernoulliMixture(centers::AbstractArray)
    log_p, p = @jit compute_mixture_params(centers)
    BernoulliMixture(log_p, p)
end


struct ConditionedBernoulliMixture{T<:Real,X,M<:AbstractVector{Bool},W}
    r::BernoulliMixture{T}
    xₛ::X
    mask::M
    w::W
end

function condition(r::BernoulliMixture, xₛ::AbstractVector, known_ii::SBitSet)
    idim = length(xₛ)
    mask = fill(false, idim)
    for i in known_ii
        mask[i] = true
    end
    condition(r, xₛ, mask)
end

function condition(r::BernoulliMixture, xₛ::AbstractVector, mask::Vector{Bool})
    _xₛ = vcat((xₛ .≤ 0)', (xₛ .> 0)')
    pzx = softmax(vec(sum(mask' .* _xₛ .* r.log_p, dims = (1,2))))
    w = StatsBase.Weights(pzx)
    ConditionedBernoulliMixture(r, xₛ, mask, w)
end

"""
    sample_all(r::AbstractSampler, n::Integer)

    Sample `n` samples including fixed (condition) part, which is 
    copied
"""
function sample_all(r::ConditionedBernoulliMixture, n::Integer)
    u = similar(r.xₛ, length(r.xₛ), n)
    sample_all!(u, r)
end

"""
    sample_all!(u, r::AbstractSampler)

    Sample `size(u,2)` samples including fixed (condition) part, which is 
    copied
"""
#INITIAN: function sample_all!(u, r::ConditionedBernoulliMixture)
function sample_all!(u::Matrix, r::ConditionedBernoulliMixture)
    mask = r.mask
    p = r.r.p
    size(u,1) != length(mask) && error("dimension of u does not match the dimension of the sampler")
    for col in axes(u, 2)
        cid = sample(r.w)
        for row in axes(u, 1)
            u[row, col] = mask[row] ? r.xₛ[row] : 2*(rand() < p[row, cid]) - 1
        end
    end
    u
end
