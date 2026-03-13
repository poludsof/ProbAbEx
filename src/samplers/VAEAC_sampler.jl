
struct VAEACSampler{M,P,S,D}
    model::M
    ps::P
    st::S
    dev::D
end

struct ConditionedVAEAC{S,X,M<:AbstractVector{Bool}}
    r::S
    xₛ::X
    mask::M
end

function load_vaeac_sampler(path, dev)
    m, ps, st = deserialize(path)
    ps = ps |> dev
    st = Lux.testmode(st) |> dev
    VAEACSampler(m, ps, st, dev)
end

function condition(r::VAEACSampler, xₛ::AbstractVector, known_ii::SBitSet)
    idim = length(xₛ)
    idim == 28*28 || error("VAEAC sampler expects idim = 784")
    mask = fill(false, idim)
    for i in known_ii
        mask[i] = true
    end
    ConditionedVAEAC(r, Float32.(xₛ), mask)
end

condition(r::VAEACSampler, xₛ::AbstractVector, mask::Vector{Bool}) =
    ConditionedVAEAC(r, Float32.(xₛ), mask)

function sample_all(r::ConditionedVAEAC, n::Integer; binary=true)
    # println("Sampling $n samples with VAEAC sampler...")
    xₛ = clamp.(r.xₛ, 0f0, 1f0)
    maskb = r.mask
    idim = length(xₛ)
    idim == 28*28 || error("VAEAC sampler expects idim = 784")

    x_img_cpu = reshape(xₛ, 28, 28)
    m2_cpu = reshape(Float32.(maskb), 28, 28)

    x01_d = reshape(xₛ, 28, 28, 1, 1) |> r.r.dev
    m_d   = reshape(Float32.(maskb), 28, 28, 1, 1) |> r.r.dev

    x01_B = x01_d .* ones(Float32, 1, 1, 1, n) |> r.r.dev
    m_B   = m_d   .* ones(Float32, 1, 1, 1, n) |> r.r.dev
    x_masked = x01_B .* m_B

    # prior_in = cat(x_masked, m_B, dims=3)
    e1 = reshape(Float32[1, 0], 1, 1, 2, 1) |> r.r.dev
    e2 = reshape(Float32[0, 1], 1, 1, 2, 1) |> r.r.dev

    prior_in = x_masked .* e1 .+ m_B .* e2
    prior_out, _ = Lux.apply(r.r.model.prior, prior_in, r.r.ps.prior, r.r.st.prior)

    ldim = r.r.model.ldim
    μp = @view prior_out[1:ldim, :]
    logσp = clamp.(@view(prior_out[ldim+1:end, :]), -5f0, 3f0)

    ε = (randn(Float32, size(μp))) #|> r.r.dev)
    z = μp .+ exp.(logσp) .* ε

    z_reshaped = reshape(z, 1, 1, ldim, n)
    z_tiled = repeat(z_reshaped, 28, 28, 1, 1)

    dec_in = cat(x_masked, m_B, z_tiled, dims=3)
    logits_spatial, _ = Lux.apply(r.r.model.decoder, dec_in, r.r.ps.decoder, r.r.st.decoder)

    probs = Lux.sigmoid.(logits_spatial)
    probs_cpu = reshape(Array(probs), 28, 28, n)

    gen = binary ? Float32.(rand(Float32, 28, 28, n) .< probs_cpu) : Float32.(probs_cpu)

    out = similar(gen)
    miss = 1f0 .- m2_cpu
    for k in 1:n
        out[:, :, k] .= x_img_cpu .* m2_cpu .+ gen[:, :, k] .* miss
    end
    out
end



function sample_all(r::ConditionedVAEAC, n::Integer; binary=true)
    extend_i = ones(Float32, 1, 1, 1, n) |> r.r.dev
    xₛ = r.xₛ|> r.r.dev
    maskb = Float32.(r.mask) |> r.r.dev
    idim = length(xₛ)
    idim == 28*28 || error("VAEAC sampler expects idim = 784")

    # prior_in = cat(x_masked, m_B, dims=3)
    e1 = reshape(Float32[1, 0], 1, 1, 2, 1) |> r.r.dev
    e2 = reshape(Float32[0, 1], 1, 1, 2, 1) |> r.r.dev
    ε = randn(Float32, r.r.model.prior.layers[end].out_dims, n) |> r.r.dev

    probs = sample_all_do(xₛ, maskb, extend_i, e1, e2, ε, r)

    probs_cpu = reshape(Array(probs), 28, 28, n)
    gen = binary ? Float32.(rand(Float32, 28, 28, n) .< probs_cpu) : Float32.(probs_cpu)
    
    x_img_cpu = reshape(xₛ, 28, 28)

    out = similar(gen)
    miss = 1f0 .- m2_cpu
    for k in 1:n
        out[:, :, k] .= x_img_cpu .* m2_cpu .+ gen[:, :, k] .* miss
    end
    out

end

function sample_all_do(xₛ, maskb, extend_i, e1, e2, ε, r)
    # println("Sampling $n samples with VAEAC sampler...")
    xₛ = clamp.(xₛ, 0f0, 1f0)
    x01_d = reshape(xₛ, 28, 28, 1, 1) 
    m_d   = reshape(maskb, 28, 28, 1, 1) 

    x01_B = x01_d .* extend_i
    m_B   = m_d   .* extend_i
    x_masked = x01_B .* m_B

    prior_in = x_masked .* e1 .+ m_B .* e2
    prior_out, _ = Lux.apply(r.r.model.prior, prior_in, r.r.ps.prior, r.r.st.prior)

    ldim = r.r.model.ldim
    μp = @view prior_out[1:ldim, :]
    logσp = clamp.(@view(prior_out[ldim+1:end, :]), -5f0, 3f0)

    z = μp .+ exp.(logσp) .* ε

    z_reshaped = reshape(z, 1, 1, ldim, n)
    z_tiled = repeat(z_reshaped, 28, 28, 1, 1)

    dec_in = cat(x_masked, m_B, z_tiled, dims=3)
    logits_spatial, _ = Lux.apply(r.r.model.decoder, dec_in, r.r.ps.decoder, r.r.st.decoder)

    probs = Lux.sigmoid.(logits_spatial)
end

function sample_all_core(prior, decoder, ldim, x01_B, m_B, ε, u, ps_prior, st_prior, ps_dec, st_dec, binary::Bool)
    # println("Sampling $n samples with VAEAC sampler...")
    x_masked = x01_B .* m_B

    e_x2 = reshape(Float32[1, 0], 1, 1, 2, 1)
    e_m2 = reshape(Float32[0, 1], 1, 1, 2, 1)
    prior_in = x_masked .* e_x2 .+ m_B .* e_m2

    prior_out, _ = Lux.apply(prior, prior_in, ps_prior, st_prior)

    Sμ = zeros(Float32, ldim, 2ldim)
    for i in 1:ldim
        Sμ[i, i] = 1f0
    end
    Sσ = zeros(Float32, ldim, 2ldim)
    for i in 1:ldim
        Sσ[i, ldim + i] = 1f0
    end

    μp    = Sμ * prior_out
    logσp = clamp.(Sσ * prior_out, -5f0, 3f0)

    z = μp .+ exp.(logσp) .* ε

    onesHW = ones(Float32, 28, 28, 1, 1)
    z_reshaped = reshape(z, 1, 1, ldim, size(z, 2))
    z_tiled = onesHW .* z_reshaped

    C = 2 + ldim
    e_x = reshape(vcat(1f0, zeros(Float32, C - 1)), 1, 1, C, 1)
    e_m = reshape(vcat(0f0, 1f0, zeros(Float32, C - 2)), 1, 1, C, 1) 

    E = zeros(Float32, C, ldim)
    for j in 1:ldim
        E[2 + j, j] = 1f0
    end
    E = reshape(E, 1, 1, C, ldim, 1)

    z5 = reshape(z_tiled, 28, 28, 1, ldim, size(z, 2)) 
    z_C = sum(E .* z5, dims=4)
    z_C = reshape(z_C, 28, 28, C, size(z, 2)) 

    dec_in = x_masked .* e_x .+ m_B .* e_m .+ z_C
    logits_spatial, _ = Lux.apply(decoder, dec_in, ps_dec, st_dec)

    probs = Lux.sigmoid.(logits_spatial)

    gen = binary ? Float32.(u .< probs) : Float32.(probs) 

    out = x01_B .* m_B .+ gen .* (1f0 .- m_B) 
    out
end

function sample_all(r::ConditionedVAEAC, compiled_sample, n::Integer; binary=true)
    dev = r.r.dev
    ldim = r.r.model.ldim

    xₛ = clamp.(r.xₛ, 0f0, 1f0)
    maskb = r.mask

    x01 = reshape(xₛ, 28, 28, 1, 1) |> dev
    m1  = reshape(Float32.(maskb), 28, 28, 1, 1) |> dev

    onesB = ones(Float32, 1, 1, 1, n) |> dev
    x01_B = x01 .* onesB
    m_B   = m1  .* onesB

    ε = randn(Float32, ldim, n) |> dev
    u = rand(Float32, 28, 28, 1, n) |> dev

    compiled_sample(r.r.model.prior,
                    r.r.model.decoder,
                    ldim,
                    x01_B,
                    m_B,
                    ε,
                    u,
                    r.r.ps.prior,
                    r.r.st.prior,
                    r.r.ps.decoder,
                    r.r.st.decoder,
                    binary)
end

function accuracy_sdp_core( prior, decoder, cls_model,
                            x01, m1, ε, u,
                            ps_prior, st_prior,
                            ps_dec, st_dec,
                            ps_cls, st_cls,
                            y::Int, ldim::Int, binary::Bool )
    n = size(ε, 2)

    onesB = ones(Float32, 1, 1, 1, n)
    x01_B = x01 .* onesB
    m_B   = m1  .* onesB

    x_masked = x01_B .* m_B

    e_x2 = reshape(Float32[1, 0], 1, 1, 2, 1)
    e_m2 = reshape(Float32[0, 1], 1, 1, 2, 1)
    prior_in = x_masked .* e_x2 .+ m_B .* e_m2 # cat(x_masked, m_B); number of channels = 2 (masked image + mask)

    prior_out, _ = Lux.apply(prior, prior_in, ps_prior, st_prior)

    # split prior output into mean and log std
    Sμ = zeros(Float32, ldim, 2ldim)
    Sσ = zeros(Float32, ldim, 2ldim)
    for i in 1:ldim
        Sμ[i, i] = 1f0
        Sσ[i, ldim + i] = 1f0
    end
    μp    = Sμ * prior_out
    logσp = clamp.(Sσ * prior_out, -5f0, 3f0)

    z = μp .+ exp.(logσp) .* ε # reparameterization trick - sample z from q(z|x_masked) using μp and logσp, shape of z is (ldim, n)

    onesHW = ones(Float32, 28, 28, 1, 1)
    z_reshaped = reshape(z, 1, 1, ldim, n)
    z_tiled = onesHW .* z_reshaped # (28, 28, ldim, n)

    C = 2 + ldim # number of channels for decoder input: masked image, mask, and latent z
    e_x = reshape(vcat(1f0, zeros(Float32, C - 1)), 1, 1, C, 1)  # one-hot encoding for masked image channel
    e_m = reshape(vcat(0f0, 1f0, zeros(Float32, C - 2)), 1, 1, C, 1) # one-hot encoding for mask channel

    # one-hot encoding for latent z channels
    E = zeros(Float32, C, ldim)
    for j in 1:ldim
        E[2 + j, j] = 1f0
    end
    E = reshape(E, 1, 1, C, ldim, 1)

    # reshape and tile z to match the spatial dimensions of the decoder input, then use E to select the appropriate channels for z in the decoder input
    z5 = reshape(z_tiled, 28, 28, 1, ldim, n)
    z_C = sum(E .* z5, dims=4)
    z_C = reshape(z_C, 28, 28, C, n)

    dec_in = x_masked .* e_x .+ m_B .* e_m .+ z_C # combine masked image, mask, and latent z into decoder input
    logits_spatial, _ = Lux.apply(decoder, dec_in, ps_dec, st_dec)
    probs = Lux.sigmoid.(logits_spatial)

    gen = binary ? Float32.(u .< probs) : Float32.(probs) # optionally sample non-binary image
    x = x01_B .* m_B .+ gen .* (1f0 .- m_B) # combine known pixels from x01_B with generated pixels from generator

    scores, _ = Lux.apply(cls_model, x, ps_cls, st_cls)

    preds = mod1.(vec(argmax(scores, dims=1)), size(scores, 1))
    correct = Float32.(preds .== y)
    return sum(correct) / Float32(n)
end

function build_compiled_accuracy(sampler, model_cls, ps_cls, st_cls, dev, num_samples)
    ldim = sampler.model.ldim

    x01_0 = zeros(Float32, 28, 28, 1, 1) |> dev # one example image
    m1_0  = zeros(Float32, 28, 28, 1, 1) |> dev # one example mask
    ε0    = zeros(Float32, ldim, num_samples) |> dev # (random) latent noise for sampling
    u0    = zeros(Float32, 28, 28, 1, num_samples) |> dev # 

    Reactant.@compile accuracy_sdp_core(
        sampler.model.prior,
        sampler.model.decoder,
        model_cls,
        x01_0, m1_0, ε0, u0,
        sampler.ps.prior, sampler.st.prior,
        sampler.ps.decoder, sampler.st.decoder,
        ps_cls, st_cls,
        1, ldim, true
    )
end

function accuracy_sdp(ii::SBitSet, sm, sampler, num_samples, compiled_acc,
                      model_cls, ps_cls, st_cls; verbose=false)
    mask = fill(false, length(sm.input))
    for i in ii
        mask[i] = true
    end

    dev = sampler.dev
    x01 = reshape(clamp.(Float32.(sm.input), 0f0, 1f0), 28, 28, 1, 1) |> dev
    m1  = reshape(Float32.(mask), 28, 28, 1, 1) |> dev

    ldim = sampler.model.ldim
    ε = randn(Float32, ldim, num_samples) |> dev
    u = rand(Float32, 28, 28, 1, num_samples) |> dev

    acc = compiled_acc(
        sampler.model.prior,
        sampler.model.decoder,
        model_cls,
        x01, m1, ε, u,
        sampler.ps.prior, sampler.st.prior,
        sampler.ps.decoder, sampler.st.decoder,
        ps_cls, st_cls,
        sm.output, ldim, true
    )
    verbose && println("accuracy = ", acc)
    acc
end

function accuracy_sdp_batched(ii::SBitSet, sm, sampler, num_samples, compiled_acc, model_cls, ps_cls, st_cls; batch_size=1000)
    n_batches = div(num_samples, batch_size)
    total_acc = 0f0

    for _ in 1:n_batches
        acc = accuracy_sdp(ii, sm, sampler, batch_size, compiled_acc, model_cls, ps_cls, st_cls)
        total_acc += acc
    end

    return total_acc / n_batches
end