import ProbAbEx as PAE
# using Reactant, Random, Lux, MLUtils, StatsBase, Optimisers, OneHotArrays, Serialization

ts_mnist = PAE.train_mnist(nepochs = 10)
ts_mnist = ts_mnist |> PAE.cpu_device() 
serialize("models/mnist_conv_model.jls", (model=ts_mnist.model, ps=ts_mnist.parameters, st=ts_mnist.states))

data = open(deserialize, "models/mnist_conv_model.jls")
ts_loaded = Lux.Training.TrainState(data.model, data.ps, data.st, Adam(3.0f-4))

#test model
train_acc = PAE.accuracy(ts_loaded.model, ts_loaded.parameters, ts_loaded.states, PAE.load_mnist(128; N=60000)[1])
test_acc  = PAE.accuracy(ts_loaded.model, ts_loaded.parameters, ts_loaded.states, PAE.load_mnist(128; N=60000)[2])
println("Loaded model - train acc = $(round(train_acc, digits=4)), test acc = $(round(test_acc, digits=4))")