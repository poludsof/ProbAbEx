
struct VAEACSampler{M,P,S}
    model::M
    ps::P
    st::S
end

function VAEACSampler(train_state)
    model, ps, st = train_state.model, train_state.parameters, train_state.states
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
    n = size(u, 2)
    size(u, 1) == length(mask) || error("dimension of u does not match the dimension of the sampler")

    m = r.mask
    x  = repeat(reshape(Float32.(r.xₛ), length(r.xₛ), 1), 1, n)
    Mcol = Float32.(mk)
    m  = repeat(reshape(Mcol, length(r.xₛ), 1), 1, n)

    p = impute!(r.r.model, r.r.ps, r.r.st, X, M)

    @inbounds for j in 1:n
        for i in 1:length(r.xₛ)
            u[i, j] = mk[i] ? r.xₛ[i] : (rand() < P[i, j] ? 1 : -1)
        end
    end
    return u
end

function impute!(model, ps, st, x::AbstractMatrix, m::AbstractMatrix)
    ldim = getfield(model, :ldim)
    ε = randn(Float32, ldim, size(x, 2))
    (logits, μq, logσq, μp, logσp), _ = Lux.apply(model, (x, mask, ε), ps, st)
    return σ.(logits)
end

sampler = VAEACSampler(deserialize("models/mnist_vaeac_model_20.jls"))

# x = PAE.load_binary_mnist_matrix()[:, 2]
# mask = random_mask(5; D=784)

