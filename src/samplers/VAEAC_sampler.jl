
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

# function sample_all(r::ConditionedVAEAC, n::Integer; binary=true)
#     # println("Sampling $n samples with VAEAC sampler...")
#     xₛ = clamp.(r.xₛ, 0f0, 1f0)
#     maskb = r.mask
#     idim = length(xₛ)
#     idim == 28*28 || error("VAEAC sampler expects idim = 784")

#     x_img_cpu = reshape(xₛ, 28, 28)
#     m2_cpu = reshape(Float32.(maskb), 28, 28)

#     x01_d = reshape(xₛ, 28, 28, 1, 1) |> r.r.dev
#     m_d   = reshape(Float32.(maskb), 28, 28, 1, 1) |> r.r.dev

#     x01_B = x01_d .* ones(Float32, 1, 1, 1, n) |> r.r.dev
#     m_B   = m_d   .* ones(Float32, 1, 1, 1, n) |> r.r.dev
#     x_masked = x01_B .* m_B

#     # prior_in = cat(x_masked, m_B, dims=3)
#     e1 = reshape(Float32[1, 0], 1, 1, 2, 1) |> r.r.dev
#     e2 = reshape(Float32[0, 1], 1, 1, 2, 1) |> r.r.dev

#     prior_in = x_masked .* e1 .+ m_B .* e2
#     prior_out, _ = Lux.apply(r.r.model.prior, prior_in, r.r.ps.prior, r.r.st.prior)

#     ldim = r.r.model.ldim
#     μp = @view prior_out[1:ldim, :]
#     logσp = clamp.(@view(prior_out[ldim+1:end, :]), -5f0, 3f0)

#     ε = (randn(Float32, size(μp))) #|> r.r.dev)
#     z = μp .+ exp.(logσp) .* ε

#     z_reshaped = reshape(z, 1, 1, ldim, n)
#     z_tiled = repeat(z_reshaped, 28, 28, 1, 1)

#     dec_in = cat(x_masked, m_B, z_tiled, dims=3)
#     logits_spatial, _ = Lux.apply(r.r.model.decoder, dec_in, r.r.ps.decoder, r.r.st.decoder)

#     probs = Lux.sigmoid.(logits_spatial)
#     probs_cpu = reshape(Array(probs), 28, 28, n)

#     gen = binary ? Float32.(rand(Float32, 28, 28, n) .< probs_cpu) : Float32.(probs_cpu)

#     out = similar(gen)
#     miss = 1f0 .- m2_cpu
#     for k in 1:n
#         out[:, :, k] .= x_img_cpu .* m2_cpu .+ gen[:, :, k] .* miss
#     end
#     out
# end

function sample_all_core(prior, decoder, ldim, x01_B, m_B, ε, u, ps_prior, st_prior, ps_dec, st_dec, binary::Bool)
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
    reshape(out, 28, 28, size(out, 4))
end

function sample_all_compiled(r::ConditionedVAEAC, compiled_sample, n::Integer; binary=true)
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

    compiled_sample(
        r.r.model.prior,
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
        binary
    )
end