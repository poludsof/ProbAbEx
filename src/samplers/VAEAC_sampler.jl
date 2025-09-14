
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


""" Example usage: """ #! todo: delete

# sampler = VAEACSampler(deserialize("models/mnist_vaeac_model_20.jls"))

# x = load_binary_mnist_matrix()[:, 2]
# random_mask(n; D=784, rng=Random.default_rng()) = (m = falses(D); m[view(randperm(rng, D), 1:n)] .= true; m )
# mask = random_mask(30; D=784)
# mask = collect(mask)
# N = 10
# r = condition(sampler, x, mask)
# u = sample_all(r, N)

# for i_sample in 1:N
#     g = reshape(u[:, i_sample], 28, 28)
#     m = reshape(mask[:, 1], 28, 28)
#     img = Array{RGB{Float32}}(undef, 28, 28)

#     for i in 1:28, j in 1:28
#         if m[i, j] == 1
#             img[i, j] = RGB{Float32}(0.5, 0, 0)
#             if x[(j-1)*28 + i, 1] == 1
#                 img[i, j] = RGB{Float32}(1, 0, 0)
#             end
#         else
#             gray_val = clamp(g[i, j], 0, 1)
#             img[i, j] = RGB{Float32}(gray_val, gray_val, gray_val)
#         end
#     end
#     display(reverse(Base.rotr90(img), dims=2))
# end
