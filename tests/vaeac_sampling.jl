
function sample_and_save_corrected(x_cpu, mask_cpu, model, ps, st; binary=true, filename="vaeac_result.png")
    st = Lux.testmode(st)
    
    x_masked = x_cpu .* mask_cpu
    
    x_masked_img = reshape(x_masked, 28, 28, 1, 1)
    mask_img     = reshape(mask_cpu, 28, 28, 1, 1)
    
    prior_in = cat(x_masked_img, mask_img, dims=3)
    
    prior_out, _ = Lux.apply(model.prior, prior_in, ps.prior, st.prior)
    
    μp = @view prior_out[1:model.ldim, :]
    logσp = clamp.(prior_out[model.ldim+1:end, :], -10f0, 10f0)
    
    ε = randn(Float32, size(μp))
    z = μp .+ exp.(logσp) .* ε
    
    z_reshaped = reshape(z, 1, 1, model.ldim, 1)
    z_tiled = repeat(z_reshaped, 28, 28, 1, 1)
    
    dec_in = cat(x_masked_img, mask_img, z_tiled, dims=3)
    
    logits_spatial, _ = Lux.apply(model.decoder, dec_in, ps.decoder, st.decoder)
    probs = Lux.sigmoid.(logits_spatial)

    probs_flat = reshape(probs, :, 1)
    
    if binary
        generated_pixels = Float32.(probs_flat .> 0.5f0)
    else
        generated_pixels = probs_flat
    end

    reconstruction = x_masked .+ generated_pixels .* (1f0 .- mask_cpu)

    to_mat(flat) = rotr90(reshape(flat, 28, 28)')
    fig = Figure(resolution = (900, 350), backgroundcolor = :white)
    
    ax1 = Axis(fig[1, 1], title = "Original")
    hidedecorations!(ax1)
    image!(ax1, to_mat(x_cpu), colormap = :grays)
    
    ax2 = Axis(fig[1, 2], title = "Masked Input")
    hidedecorations!(ax2)
    
    img_mat = to_mat(x_cpu)
    mask_mat = to_mat(mask_cpu)
    colored_input = [mask_mat[i,j] > 0.5 ? (img_mat[i,j] > 0.5 ? RGB(1.0, 0.2, 0.2) : RGB(0.4, 0.0, 0.0)) : Gray(0.5) for i in 1:28, j in 1:28]
    image!(ax2, colored_input)
    
    ax3 = Axis(fig[1, 3], title = binary ? "Binary Reconstruction" : "Reconstruction")
    hidedecorations!(ax3)
    image!(ax3, to_mat(reconstruction), colormap = :grays)
    
    save(filename, fig)
    @info "Saved result to $filename"
    
    return reconstruction
end

to_cpu(x) = x
to_cpu(x::AbstractArray) = Array(x)
to_cpu(nt::NamedTuple) = NamedTuple{keys(nt)}(map(to_cpu, values(nt)))
to_cpu(t::Tuple) = map(to_cpu, t)
# to_cpu(ts) = Lux.Training.TrainState(ts.model, to_cpu(ts.parameters), to_cpu(ts.states), ts.optimizer)

ps = to_cpu(ts.parameters)
st = to_cpu(ts.states)
model = ts.model

x_raw = PAE.load_binary_mnist_matrix()[:, 3]
x_cpu = reshape(Float32.(x_raw), :, 1)

mask_raw = rand(Float32, 784) .> 0.8
mask = reshape(Float32.(mask_raw), :, 1)

sample_and_save_corrected(x_cpu, mask, ts.model, ps, st, binary=true)