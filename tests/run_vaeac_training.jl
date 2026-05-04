# using ColorTypes: RGB
# using ColorTypes: Gray, heatmap!
# using CairoMakie: Figure, Axis, image!, hidespines!, hidedecorations!, save
import ProbAbEx as PAE
using Serialization, Random, Optimisers, Lux, Base, NNlib, MLDatasets, StaticBitSets
using Reactant
import Lux: σ
using CairoMakie
using Statistics

""" ================ Save and Load Model ================= """
# JLS
function save_vaeac_model(filename, model, ps, st)
    cdev = cpu_device()
    ps_cpu = cdev(ps)
    st_cpu = cdev(st)
    print("Saving model to $filename ... ")
    open(filename, "w") do io
        serialize(io, (model, ps_cpu, st_cpu))
    end
    println("Done!")
end

""" ========= Create and train model ========= """

Reactant.set_default_backend("gpu")
dev = reactant_device(; force=true)

ts, logs = PAE.train_vaeac(epochs=50, lr=0.001f0, batch_size=100)


# model, ps, st = deserialize(joinpath(@__DIR__, "..", "models", "test_deserialize.jls"))
# model, ps, st = deserialize("models/mnist_vaeac_conv_model.jls")
# # print fields of ts: model, parameters, states, optimizer
# println("Fields of the __: ", fieldnames(typeof(ts)))


""" ================ Save and Load Model ================= """

save_vaeac_model("models/small_new_vaeac.jls", ts.model, ts.parameters, ts.states)

# ts2 = load_vaeac_jls("models/mnist_vaeac_model_50.jls"; lr=0.001f0)


## Vizualization of the training results

using CairoMakie

function plot_training_results(logs::Vector{PAE.TrainingLog})
    epochs = [l.epoch for l in logs]
    total  = [l.avg_loss for l in logs]
    recon  = [l.avg_recon for l in logs]
    kl     = [l.avg_kl for l in logs]
    betas  = [l.β for l in logs]

    fig = Figure(size = (800, 900), fontsize = 18)

    # 1. Total Loss (Log Scale)
    ax1 = Axis(fig[1, 1], 
        title = "VAEAC Training Metrics (Log Scale)",
        ylabel = "Total Loss",
        yscale = log10, # <--- Fixes the squashing
        xticklabelsvisible = false)
    lines!(ax1, epochs, total, color = :black, linewidth = 2)

    # 2. Reconstruction Loss (Linear usually fine, but using log for consistency)
    ax2 = Axis(fig[2, 1], 
        ylabel = "Reconstruction",
        yscale = log10,
        xticklabelsvisible = false)
    lines!(ax2, epochs, recon, color = :blue, linewidth = 2)

    # 3. KL Divergence (Log Scale)
    ax3 = Axis(fig[3, 1], 
        xlabel = "Epoch", 
        ylabel = "KL Divergence",
        yscale = log10)
    
    l1 = lines!(ax3, epochs, kl .+ 1e-6, color = :red, linewidth = 2) # Added epsilon to avoid log(0)
    
    ax3_right = Axis(fig[3, 1], 
        ylabel = "Beta (β)", 
        yaxisposition = :right,
        ygridvisible = false)
    l2 = lines!(ax3_right, epochs, betas, color = :gray, linestyle = :dash, linewidth = 2)
    
    hidespines!(ax3_right)
    hidexdecorations!(ax3_right)

    linkxaxes!(ax1, ax2, ax3)

    Legend(fig[3, 1], [l1, l2], ["KL", "β"], 
           tellheight = false, tellwidth = false, 
           halign = :right, valign = :top, margin = (10, 10, 10, 10))

    return fig
end
        
mkpath("training_plots")
serialize(joinpath("training_plots", "small_training_logs.jls"), logs)
save(joinpath("training_plots", "small_training_results.png"), plot_training_results(logs))