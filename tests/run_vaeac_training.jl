# using ColorTypes: RGB
# using ColorTypes: Gray, heatmap!
# using CairoMakie: Figure, Axis, image!, hidespines!, hidedecorations!, save
import ProbAbEx as PAE
using Serialization, Random, Optimisers, Lux, Base, NNlib, MLDatasets, StaticBitSets
using Reactant
import Lux: σ

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

# ts = PAE.train_vaeac(epochs=30, lr=0.001f0, batch_size=100)

# model, ps, st = deserialize(joinpath(@__DIR__, "..", "models", "test_deserialize.jls"))
# model, ps, st = deserialize("models/mnist_vaeac_conv_model.jls")
# # print fields of ts: model, parameters, states, optimizer
# println("Fields of the __: ", fieldnames(typeof(ts)))


""" ================ Save and Load Model ================= """

# save_vaeac_model("models/use_this_vaeac.jls", ts.model, ts.parameters, ts.states)

# ts2 = load_vaeac_jls("models/mnist_vaeac_model_50.jls"; lr=0.001f0)
