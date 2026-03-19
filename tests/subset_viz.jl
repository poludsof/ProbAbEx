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

# mask_raw = rand(Float32, 784) .> 0.8 # ! random mask for testing
mask_raw = falses(784)
for idx in solution_subsets # ! use solution_subsets from forward_search to create mask
    mask_raw[idx] = true
end
mask = reshape(Float32.(mask_raw), :, 1)

x_raw = PAE.load_binary_mnist_matrix()[:, 2]
x_cpu = reshape(Float32.(x_raw), :, 1)

create_overlay_image(x_cpu, mask)

##
""" visualize original, masked, and reconstructed images in a grid """
function sample_n_reconstructions(x_cpu, mask_cpu, model, ps, st, N; filename="samples.png")
    st = Lux.testmode(st)
    
    x_masked_img = reshape(x_cpu .* mask_cpu, 28, 28, 1, 1)
    mask_img     = reshape(mask_cpu, 28, 28, 1, 1)

    samples = []

    for _ in 1:N
        # Prior pass
        prior_in = cat(x_masked_img, mask_img, dims=3)
        prior_out, _ = Lux.apply(model.prior, prior_in, ps.prior, st.prior)
        μp     = @view prior_out[1:model.ldim, :]
        logσp  = clamp.(prior_out[model.ldim+1:end, :], -5f0, 3f0)
        z      = μp .+ exp.(logσp) .* randn(Float32, size(μp))

        # Decoder pass
        z_tiled = repeat(reshape(z, 1, 1, model.ldim, 1), 28, 28, 1, 1)
        dec_in  = cat(x_masked_img, mask_img, z_tiled, dims=3)
        logits, _ = Lux.apply(model.decoder, dec_in, ps.decoder, st.decoder)
        probs   = reshape(Lux.sigmoid.(logits), :)

        # Build grayscale image for this sample
        img = Array{RGB{Float32}}(undef, 28, 28)
        idx(i, j) = (j-1)*28 + i
        for i in 1:28, j in 1:28
            k = idx(i, j)
            val = clamp(probs[k], 0f0, 1f0)
            img[i, j] = RGB{Float32}(val, val, val)
        end
        push!(samples, img)
    end

    # Plot all N samples in a row
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

to_cpu(x) = x
to_cpu(x::AbstractArray) = Array(x)
to_cpu(nt::NamedTuple) = NamedTuple{keys(nt)}(map(to_cpu, values(nt)))
to_cpu(t::Tuple) = map(to_cpu, t)

model, ps, st = deserialize(joinpath(@__DIR__, "..", "models", "use_this_vaeac.jls"))

ps = to_cpu(ps)
st = to_cpu(st)
model = to_cpu(model)

# samples = sample_n_reconstructions(x_cpu, mask, model, ps, st, 5)
