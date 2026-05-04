using CairoMakie
import ProbAbEx as PAE
using Serialization
using ColorTypes

function create_overlay_image(x_cpu, mask_cpu)
    img = Array{RGB{Float32}}(undef, 28, 28)
    idx(i, j) = (j-1)*28 + i

    for i in 1:28, j in 1:28
        k = idx(i, j)
        if mask_cpu[k] > 0.5
            # Observed pixel → Red
            if x_cpu[k] > 0.5
                img[i, j] = RGB{Float32}(1.0, 0.0, 0.0)  # Bright Red (ink)
            else
                img[i, j] = RGB{Float32}(0.3, 0.0, 0.0)  # Dark Red (background)
            end
        else
            # Not observed → normal grayscale
            val = x_cpu[k]
            img[i, j] = RGB{Float32}(val, val, val)
        end
    end

    function style_axis!(ax, title_str, img_data)
        empty!(ax)
        ax.title = title_str
        ax.titlecolor = :black
        ax.xgridvisible = false; ax.ygridvisible = false
        ax.xticksvisible = false; ax.yticksvisible = false
        ax.xticklabelsvisible = false; ax.yticklabelsvisible = false
        ax.backgroundcolor = :black
        
        image!(ax, 0..28, 0..28, img_data; interpolate = false)
        
        for k in 0:28
            hlines!(ax, k; color = :gray20, linewidth = 1)
            vlines!(ax, k; color = :gray20, linewidth = 1)
        end
        
        xlims!(ax, 0, 28)
        ylims!(ax, 28, 0) # Flip Y to match standard image coords
    end

    fig = Figure(resolution = (450, 450), backgroundcolor = :white)
    ax = Axis(fig[1, 1])

    style_axis!(ax, "Digit + Mask Overlay", img)
    save("overlay.png", fig)
end

mask_raw = rand(Float32, 784) .> 0.95 # ! random mask for testing
# mask_raw = falses(784)
# for idx in solution_subsets # ! use solution_subsets from forward_search to create mask
#     mask_raw[idx] = true
# end
mask = reshape(Float32.(mask_raw), :, 1)

x_raw = PAE.load_binary_mnist_matrix()[:, 1]
x_cpu = reshape(Float32.(x_raw), :, 1)

# create_overlay_image(x_cpu, mask)

##
""" visualize original, masked, and reconstructed images in a grid """
function sample_n_reconstructions(x_cpu, mask_cpu, model, ps, st, N; filename="samples.png", binarize::Bool=false)
    st = Lux.testmode(st)
    
    x_masked_img = reshape(x_cpu .* mask_cpu, 28, 28, 1, 1)
    mask_img     = reshape(mask_cpu, 28, 28, 1, 1)

    samples = []

    for _ in 1:N
        # Prior pass
        prior_in = cat(x_masked_img, mask_img, dims=3)
        prior_out, _ = Lux.apply(model.prior, prior_in, ps.prior, st.prior)
        μp     = @view prior_out[1:model.ldim, :]
        logσp  = clamp.(prior_out[model.ldim+1:end, :], -10f0, 10f0)
        z      = μp .+ exp.(logσp) .* randn(Float32, size(μp))

        # Decoder pass
        # z_tiled = repeat(reshape(z, 1, 1, model.ldim, 1), 28, 28, 1, 1)
        # dec_in  = cat(x_masked_img, mask_img, z_tiled, dims=3)
        # logits, _ = Lux.apply(model.decoder, dec_in, ps.decoder, st.decoder)
        # probs   = reshape(Lux.sigmoid.(logits), :)


        dec_in = vcat(z, x_cpu .* mask_cpu, mask_cpu)
        logits, _ = Lux.apply(model.decoder, dec_in, ps.decoder, st.decoder)
        probs   = reshape(Lux.sigmoid.(logits), :)

        img = Array{RGB{Float32}}(undef, 28, 28)
        idx(i, j) = (j-1)*28 + i
        for i in 1:28, j in 1:28
            k = idx(i, j)
            if mask_cpu[k] > 0.5
                if x_cpu[k] > 0.5
                    img[i, j] = RGB{Float32}(1.0, 0.0, 0.0)
                else
                    img[i, j] = RGB{Float32}(0.3, 0.0, 0.0)
                end
            else
                val = clamp(probs[k], 0f0, 1f0)
                val = binarize ? (val > 0.5f0 ? 1f0 : 0f0) : val
                img[i, j] = RGB{Float32}(val, val, val)
            end
        end
        push!(samples, img)
    end

    fig = Figure(resolution = (450 * N, 450), backgroundcolor = :black)
    for n in 1:N
        ax = Axis(fig[1, n])
        ax.xgridvisible = false; ax.ygridvisible = false
        ax.xticksvisible = false; ax.yticksvisible = false
        ax.xticklabelsvisible = false; ax.yticklabelsvisible = false
        ax.backgroundcolor = :black
        ax.leftspinevisible = false; ax.rightspinevisible = false
        ax.topspinevisible = false; ax.bottomspinevisible = false

        image!(ax, 0..28, 0..28, samples[n]; interpolate = false)
        xlims!(ax, 0, 28); ylims!(ax, 28, 0)
    end

    save(filename, fig)
    @info "Saved $N samples to $filename"
    return samples
end

function reconstruct_with_proposal(x_cpu, mask_cpu, model, ps, st)
    st = Lux.testmode(st)

    x_img = reshape(x_cpu, 28, 28, 1, 1)
    mask_img = reshape(mask_cpu, 28, 28, 1, 1)
    x_masked_img = x_img .* mask_img

    prop_in = cat(x_img, mask_img, dims=3)
    prop_out, _ = Lux.apply(model.proposal, prop_in, ps.proposal, st.proposal)

    μq = @view prop_out[1:model.ldim, :]
    z = μq

    z_tiled = repeat(reshape(z, 1, 1, model.ldim, 1), 28, 28, 1, 1)
    dec_in = cat(x_masked_img, mask_img, z_tiled, dims=3)

    logits, _ = Lux.apply(model.decoder, dec_in, ps.decoder, st.decoder)
    probs = reshape(Lux.sigmoid.(logits), :)

    @show mean(probs)
    @show mean(probs[mask_cpu .< 0.5])
    @show minimum(probs)
    @show maximum(probs)

    return probs
end
function check_reconstruction_quality(probs, x_cpu, mask_cpu)
    unknown = mask_cpu .< 0.5f0

    y = x_cpu[unknown]
    p = probs[unknown]
    pred = Float32.(p .> 0.5f0)

    @show mean(y)
    @show mean(p)
    @show mean(pred)
    @show mean(pred .== y)

    pos = y .> 0.5f0
    neg = y .< 0.5f0

    @show mean(p[pos])
    @show mean(p[neg])

    tp = sum((pred .== 1f0) .& (y .== 1f0))
    fp = sum((pred .== 1f0) .& (y .== 0f0))
    fn = sum((pred .== 0f0) .& (y .== 1f0))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    @show precision
    @show recall
end
function reconstruct_with_prior_mean(x_cpu, mask_cpu, model, ps, st)
    st = Lux.testmode(st)

    x_img = reshape(x_cpu, 28, 28, 1, 1)
    mask_img = reshape(mask_cpu, 28, 28, 1, 1)
    x_masked_img = x_img .* mask_img

    prior_in = cat(x_masked_img, mask_img, dims=3)
    prior_out, _ = Lux.apply(model.prior, prior_in, ps.prior, st.prior)

    μp = @view prior_out[1:model.ldim, :]
    z = μp

    z_tiled = repeat(reshape(z, 1, 1, model.ldim, 1), 28, 28, 1, 1)
    dec_in = cat(x_masked_img, mask_img, z_tiled, dims=3)

    logits, _ = Lux.apply(model.decoder, dec_in, ps.decoder, st.decoder)
    probs = reshape(Lux.sigmoid.(logits), :)

    return probs
end

to_cpu(x) = x
to_cpu(x::AbstractArray) = Array(x)
to_cpu(nt::NamedTuple) = NamedTuple{keys(nt)}(map(to_cpu, values(nt)))
to_cpu(t::Tuple) = map(to_cpu, t)

model, ps, st = deserialize(joinpath(@__DIR__, "..", "models", "small_new_vaeac.jls"))

ps = to_cpu(ps)
st = to_cpu(st)
model = to_cpu(model)

samples = sample_n_reconstructions(x_cpu, mask, model, ps, st, 5)


# probs_q = reconstruct_with_proposal(x_cpu, mask, model, ps, st)
# probs_p = reconstruct_with_prior_mean(x_cpu, mask, model, ps, st)

# println("proposal:")
# check_reconstruction_quality(probs_q, x_cpu, mask)

# println("prior:")
# check_reconstruction_quality(probs_p, x_cpu, mask)



## 

struct ExperimentResult
    image_idx::Int
    threshold::Float64
    solution::Any          # the SBitSet subset (or nothing if not found)
    subset_size::Int
    time_seconds::Float64
    found::Bool
end


function create_overlay_image_to_file(x_cpu, mask_cpu, filepath; title_str="Digit + Mask Overlay")
    img = Array{RGB{Float32}}(undef, 28, 28)
    idx(i, j) = (j-1)*28 + i

    for i in 1:28, j in 1:28
        k = idx(i, j)
        if mask_cpu[k] > 0.5
            if x_cpu[k] > 0.5
                img[i, j] = RGB{Float32}(1.0, 0.0, 0.0)
            else
                img[i, j] = RGB{Float32}(0.3, 0.0, 0.0)
            end
        else
            val = x_cpu[k]
            img[i, j] = RGB{Float32}(val, val, val)
        end
    end

    fig = Figure(resolution = (450, 450), backgroundcolor = :white)
    ax  = Axis(fig[1, 1])

    empty!(ax)
    ax.title              = title_str
    ax.titlecolor         = :black
    ax.xgridvisible       = false; ax.ygridvisible       = false
    ax.xticksvisible      = false; ax.yticksvisible      = false
    ax.xticklabelsvisible = false; ax.yticklabelsvisible = false
    ax.backgroundcolor    = :black

    image!(ax, 0..28, 0..28, img; interpolate = false)

    for k in 0:28
        hlines!(ax, k; color = :gray20, linewidth = 1)
        vlines!(ax, k; color = :gray20, linewidth = 1)
    end

    xlims!(ax, 0, 28)
    ylims!(ax, 28, 0)

    save(filepath, fig)
end

function visualize_results(jls_path::String, all_X_binary; output_dir = "result_images")
    mkpath(output_dir)

    results = deserialize(jls_path)
    println("Loaded $(length(results)) results from $jls_path")

    for r in results
        # ── build mask from the SBitSet solution ──────────────────────────
        mask_raw = falses(784)
        if r.found && r.solution !== nothing
            for idx in r.solution          # iterate over set bits
                mask_raw[idx] = true
            end
        end
        mask_cpu = reshape(Float32.(mask_raw), :, 1)

        # ── grab the original image ────────────────────────────────────────
        x_raw    = Float32.(all_X_binary[:, r.image_idx])
        x_cpu    = reshape(x_raw, :, 1)

        # ── build a descriptive filename ───────────────────────────────────
        eps_str  = replace(string(r.threshold), "." => "p")   # 0.99 → "0p99"
        fname    = "img$(r.image_idx)_eps$(eps_str)_size$(r.subset_size).png"
        fpath    = joinpath(output_dir, fname)

        title = "img=$(r.image_idx)  ε=$(r.threshold)  |S|=$(r.subset_size)  t=$(round(r.time_seconds, digits=1))s"

        create_overlay_image_to_file(x_cpu, mask_cpu, fpath; title_str = title)
        println("  Saved: $fpath")
    end

    println("\nDone. $(length(results)) images written to '$output_dir'.")
end

X_binary = PAE.load_binary_mnist_matrix()   # (784, N)
# visualize_results("results/results_2.jls", X_binary; output_dir = "result_images")


function visualize_reconstructions(jls_path::String, all_X_binary, model, ps, st; N_samples = 5, output_dir  = "result_reconstructions")
    mkpath(output_dir)

    results = deserialize(jls_path)
    println("Loaded $(length(results)) results from $jls_path")

    st_cpu = Lux.testmode(st)

    for r in results
        # ── маска из SBitSet ───────────────────────────────────────────────
        mask_raw = falses(784)
        if r.found && r.solution !== nothing
            for idx in r.solution
                mask_raw[idx] = true
            end
        end
        mask_cpu = reshape(Float32.(mask_raw), :, 1)

        # ── исходное изображение ───────────────────────────────────────────
        x_cpu = reshape(Float32.(all_X_binary[:, r.image_idx]), :, 1)

        # ── имя файла ──────────────────────────────────────────────────────
        eps_str = replace(string(r.threshold), "." => "p")
        fname   = "img$(r.image_idx)_eps$(eps_str)_size$(r.subset_size)_samples.png"
        fpath   = joinpath(output_dir, fname)

        println("  img=$(r.image_idx)  ε=$(r.threshold)  |S|=$(r.subset_size) → $fname")
        # sample_n_reconstructions(x_cpu, mask_cpu, model, ps, st_cpu, N_samples; filename=fpath)
        sample_n_reconstructions(x_cpu, mask_cpu, model, ps, st, 5; filename=fpath, binarize=true)

    end

    println("\nDone. $(length(results)) × $N_samples reconstructions written to '$output_dir'.")
end

to_cpu(x) = x
to_cpu(x::AbstractArray) = Array(x)
to_cpu(nt::NamedTuple) = NamedTuple{keys(nt)}(map(to_cpu, values(nt)))
to_cpu(t::Tuple) = map(to_cpu, t)
model, ps, st = deserialize("models/small_new_vaeac.jls")
ps = to_cpu(ps); st = to_cpu(st); model = to_cpu(model)

X_binary = PAE.load_binary_mnist_matrix()

# visualize_reconstructions("results/results_2.jls", X_binary, model, ps, st; N_samples = 5, output_dir = "result_reconstructions")
