module CUDAExt

using ProbAbEx, CUDA
using ProbAbEx.StaticBitSets
using ProbAbEx.StatsBase
using Flux
using ProbAbEx: ConditionedUniformDistribution, UniformDistribution, condition, ConditionedBernoulliMixture

@inline function ProbAbEx.condition(r::UniformDistribution, xₛ::CuArray, known_ii::SBitSet)
    ConditionedUniformDistribution(cu(condition(r, Array(xₛ), known_ii).p))
end

@inline function ProbAbEx.sample_all(r::ConditionedUniformDistribution{<:CuVector}, n::Integer)
    x = CUDA.rand(Float32, length(r.p), n)
    _f(xᵢ, pᵢ) = 2*(xᵢ < pᵢ) - 1
    _f.(x, r.p)
end




##############
#   This implements sampling from conditioned bernoulli
##############
function ProbAbEx.condition(r::BernoulliMixture{<:Any,<:CuArray,<:CuMatrix}, xₛ::CuVector, mask::Vector{Bool})
    mask = cu(mask)
    ProbAbEx.condition(r, xₛ, mask)
end

function ProbAbEx.condition(r::BernoulliMixture{<:Any,<:CuArray,<:CuMatrix}, xₛ::CuVector, mask::CuVector{Bool})
    _xₛ = vcat((xₛ .≤ 0)', (xₛ .> 0)')
    pzx = softmax(vec(sum(mask' .* _xₛ .* r.log_p, dims = (1,2))))
    w = StatsBase.Weights(Vector(pzx))
    ConditionedBernoulliMixture(r, xₛ, mask, w)
end

function ProbAbEx.sample_all(r::ConditionedBernoulliMixture{<:Any, <:CuArray, <:CuArray}, n::Integer)
    println("Sampling $n samples from ConditionedBernoulliMixture on CUDA device...")
    mask = r.mask
    p = r.r.p
    xₛ = r.xₛ
    _f(mᵢ, xᵢ, xₛᵢ, pᵢ) = mᵢ ? Float32(xₛᵢ) : Float32((2*(xᵢ < pᵢ) - 1))
    
    x = CUDA.rand(Float32, size(p,1), n)
    cids = sample(1:length(r.w), r.w, n)
    _f.(mask, x, xₛ, p[:,cids])
end

end