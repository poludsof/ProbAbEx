
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
    D = length(r.xₛ)
    n = size(u, 2)
    size(u, 1) == D || error("dimension of u does not match the dimension of the sampler")

    mk = r.mask                      # Vector{Bool}, true = known
    X  = repeat(reshape(Float32.(r.xₛ), D, 1), 1, n)
    M  = repeat(reshape(Float32.(mk),    D, 1), 1, n)   # Float32 mask matrix for model

    p = impute!(r.r.model, r.r.ps, r.r.st, X, M)        # D×n probabilities

    # @inbounds for j in 1:n
    #     for i in 1:D
    #         if mk[i]
    #             u[i, j] = r.xₛ[i]
    #         else
    #             u[i, j] = (rand() < p[i, j]) ? 1 : -1
    #         end
    #     end
    # end
    # return u
    p .= ifelse.(p .> 0.5f0, 1, -1)
end

function impute!(model, ps, st, x::AbstractMatrix, m::AbstractMatrix)
    ldim = getfield(model, :ldim)
    ε = randn(Float32, ldim, size(x, 2))
    (logits, μq, logσq, μp, logσp), _ = Lux.apply(model, (x, m, ε), ps, st)
    return σ.(logits)
end

sampler = VAEACSampler(deserialize("models/mnist_vaeac_model_20.jls"))

x = load_binary_mnist_matrix()[:, 2]
random_mask(n; D=784, rng=Random.default_rng()) = (m = falses(D); m[view(randperm(rng, D), 1:n)] .= true; m )
mask = random_mask(5; D=784)

#BitVector to Vector{Bool}:
mask = collect(mask)

r = condition(sampler, x, mask)
u = sample_all(r, 10)
