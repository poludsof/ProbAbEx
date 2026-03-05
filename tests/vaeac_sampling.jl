import ProbAbEx as PAE
using Serialization, Random, Lux, Base, NNlib, MLDatasets, StaticBitSets
using ColorTypes
using CairoMakie

function sample_and_save_styled(x_cpu, mask_cpu, model, ps, st; binary=false, filename="vaeac_styled.png")
    # 1. PREPARE STATE
    st = Lux.testmode(st)
    
    # 2. CONV SAMPLING (Inference)
    # ----------------------------
    x_masked = x_cpu .* mask_cpu
    
    # Reshape inputs for Conv layers (28, 28, 1, 1)
    x_masked_img = reshape(x_masked, 28, 28, 1, 1)
    mask_img     = reshape(mask_cpu, 28, 28, 1, 1)
    
    # A. Prior Pass
    prior_in = cat(x_masked_img, mask_img, dims=3)
    prior_out, _ = Lux.apply(model.prior, prior_in, ps.prior, st.prior)
    
    μp = @view prior_out[1:model.ldim, :]
    logσp = clamp.(prior_out[model.ldim+1:end, :], -5f0, 3f0)
    ε = randn(Float32, size(μp))
    z = μp .+ exp.(logσp) .* ε
    
    # B. Decoder Pass (With Tiling for Conv Decoder)
    # Reshape z to (1, 1, 20, 1)
    z_reshaped = reshape(z, 1, 1, model.ldim, 1)
    # Tile z 28x28 times -> (28, 28, 20, 1)
    z_tiled = repeat(z_reshaped, 28, 28, 1, 1)
    
    dec_in = cat(x_masked_img, mask_img, z_tiled, dims=3)
    logits_spatial, _ = Lux.apply(model.decoder, dec_in, ps.decoder, st.decoder)
    probs = Lux.sigmoid.(logits_spatial)
    
    # Flatten outputs for easy indexing
    probs_flat = reshape(probs, :, 1)
    generated = binary ? Float32.(probs_flat .> 0.5f0) : probs_flat
    reconstruction = x_masked .+ generated .* (1f0 .- mask_cpu)
    # 3. IMAGE CONSTRUCTION (RGB Matrices)
    # ------------------------------------
    # We build 3 images manually using the User's Color Logic
    
    img_orig  = Array{RGB{Float32}}(undef, 28, 28)
    img_masked = Array{RGB{Float32}}(undef, 28, 28)
    img_recon = Array{RGB{Float32}}(undef, 28, 28)
    
    # Helper to access flattened arrays by (i, j)
    # Julia arrays are Column-Major: idx = (j-1)*28 + i
    idx(i, j) = (j-1)*28 + i

    for i in 1:28, j in 1:28
        k = idx(i, j)
        
        # 1. Original (Ground Truth) - Simple Grayscale
        val_orig = x_cpu[k]
        img_orig[i, j] = RGB{Float32}(val_orig, val_orig, val_orig)

        # 2. Masked Input (What model sees)
        # If Observed (mask=1): Red logic
        # If Missing (mask=0): Black (or very dark gray)
        if mask_cpu[k] > 0.5
            if x_cpu[k] > 0.5
                img_masked[i, j] = RGB{Float32}(1.0, 0.0, 0.0) # Bright Red (Ink)
            else
                img_masked[i, j] = RGB{Float32}(0.3, 0.0, 0.0) # Dark Red (Bg)
            end
        else
            img_masked[i, j] = RGB{Float32}(0.0, 0.0, 0.0) # Black for missing
        end

        # 3. Reconstruction (Result)
        # If Observed: Red logic (Copy input)
        # If Missing: Grayscale logic (Show Inpainting)
        # if mask_cpu[k] > 0.5
        #     # Keep observed pixels Red
        #     if x_cpu[k] > 0.5
        #         img_recon[i, j] = RGB{Float32}(1.0, 0.0, 0.0) 
        #     else
        #         img_recon[i, j] = RGB{Float32}(0.3, 0.0, 0.0) 
        #     end
        # else
        #     val_gen = clamp(generated[k], 0, 1)
        #     img_recon[i, j] = RGB{Float32}(val_gen, val_gen, val_gen)
        # end
        val_final = clamp(reconstruction[k], 0, 1)
        img_recon[i, j] = RGB{Float32}(val_final, val_final, val_final)
    end

    # 4. PLOTTING WITH GRID
    # ---------------------
    r_orig = img_orig
    r_masked = img_masked
    r_recon = img_recon

    fig = Figure(resolution = (1200, 450), backgroundcolor = :white)
    
    # Helper to style the axis like a pixel grid
    function style_axis!(ax, title_str, img_data)
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

    ax1 = Axis(fig[1, 1])
    style_axis!(ax1, "Original", r_orig)

    ax2 = Axis(fig[1, 2])
    style_axis!(ax2, "Masked Input", r_masked)

    ax3 = Axis(fig[1, 3])
    style_axis!(ax3, "Reconstruction", r_recon)

    save(filename, fig)
    @info "Saved styled grid visualization to $filename"
    
    return generated
end

to_cpu(x) = x
to_cpu(x::AbstractArray) = Array(x)
to_cpu(nt::NamedTuple) = NamedTuple{keys(nt)}(map(to_cpu, values(nt)))
to_cpu(t::Tuple) = map(to_cpu, t)
# to_cpu(ts) = Lux.Training.TrainState(ts.model, to_cpu(ts.parameters), to_cpu(ts.states), ts.optimizer)

model, ps, st = deserialize(joinpath(@__DIR__, "..", "models", "use_this_vaeac.jls"))

ps = to_cpu(ps)
st = to_cpu(st)
model = to_cpu(model)

x_raw = PAE.load_binary_mnist_matrix()[:, 6]
x_cpu = reshape(Float32.(x_raw), :, 1)

mask_raw = rand(Float32, 784) .> 0.8
mask = reshape(Float32.(mask_raw), :, 1)

sample_and_save_styled(x_cpu, mask, model, ps, st, binary=false, filename="test_test_test.png")