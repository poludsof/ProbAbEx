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
struct UniformDistribution
end

struct ConditionedUniformDistribution{P}
    p::P
end

function condition(r::UniformDistribution, xₛ, known_ii::SBitSet)
    ii = collect(known_ii)
    p = fill(0.5f0, length(xₛ))
    p[ii] .= (xₛ[ii] .> 0.5)
    ConditionedUniformDistribution(p)
end

"""
    sample_all(r::AbstractSampler, n::Integer)

    Sample `n` samples including fixed (condition) part, which is 
    copied
"""
function sample_all(r::ConditionedUniformDistribution, n::Integer)
    x = rand(Float32, length(r.p), n)
    @inbounds for j in 1:size(x,2)
        for i in 1:size(x,1)
            x[i,j] = 2*(x[i,j] < r.p[i]) - 1
        end
    end
    x
end