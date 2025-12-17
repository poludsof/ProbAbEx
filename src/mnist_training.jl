function load_mnist(batchsize::Int; N::Int = 60_000, train_split::Float64 = 0.9)
    dataset = MNIST(; split = :train)

    imgs        = dataset.features[:, :, 1:N]
    labels_raw  = dataset.targets[1:N]

    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehotbatch(labels_raw, 0:9)

    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at = train_split)

    train_loader = DataLoader(collect.((x_train, y_train));
                              batchsize = batchsize,
                              shuffle = true,
                              partial = false)
    test_loader  = DataLoader(collect.((x_test, y_test));
                              batchsize = batchsize,
                              shuffle = false,
                              partial = false)

    return train_loader, test_loader
end


# -----------------------------
# Model definition

# function build_model()
#     Lux.Chain(
#         Lux.FlattenLayer(),
#         Lux.Dense(784 => 400, relu),
#         Lux.Dense(400 => 400, relu),
#         Lux.Dense(400 => 10)
#     )
# end

function build_model()
    Chain(  Conv((5, 5), 1 => 6, relu), 
            MaxPool((2, 2)), 
            Conv((5, 5), 6 => 16, relu), 
            MaxPool((2, 2)), 
            FlattenLayer(3), 
            Dense(256 => 128, relu), 
            Dense(128 => 84, relu), 
            Dense(84 => 10) ) 
end

# function build_model()
#     Chain(
#         Conv((5, 5), 1 => 8, relu),
#         MaxPool((2, 2)),
#         FlattenLayer(3),
#         Dense(12 * 12 * 8 => 32, relu),
#         Dense(32 => 10)
#     )
# end

# function build_model()
#     Chain(
#         Conv((5, 5), 1 => 8, relu),
#         FlattenLayer(3),
#         Dense(24 * 24 * 8 => 64, relu),
#         Dense(64 => 10)
#     )
# end

# -----------------------------
# Evaluation helper

function accuracy(model, ps, st, dataloader)
    total_correct = 0
    total = 0

    st_eval = Lux.testmode(st)

    for (x, y) in dataloader
        ŷ, st_eval = Lux.apply(model, x, ps, st_eval)
        # Convert from device arrays to CPU plain arrays for onecold
        ŷ_cpu = Array(ŷ)
        y_cpu  = Array(y)

        pred_labels  = onecold(ŷ_cpu, 0:9)
        true_labels  = onecold(y_cpu,  0:9)

        total_correct += count(==(true), pred_labels .== true_labels)
        total += length(true_labels)
    end

    return total_correct / total
end

# -----------------------------
# Training loop

function train_mnist(; batchsize = 128,
                     nepochs    = 5,
                     train_split = 0.9,
                     N          = 60_000,
                     seed       = 0)

    model = build_model()
    rng = Random.Xoshiro(seed)
        
    Reactant.set_default_backend("gpu")
    dev = reactant_device()

    ps, st = Lux.setup(rng, model)
    ps = ps |> dev
    st = st |> dev

    println("Loading MNIST…")
    train_loader_cpu, test_loader_cpu = load_mnist(batchsize; N = N, train_split = train_split)
    train_loader_dev = DeviceIterator(dev, train_loader_cpu)


    loss_layer = CrossEntropyLoss(; logits = Val(true))
    opt = Adam(3.0f-4)

    ts = Lux.Training.TrainState(model, ps, st, opt)

    println("Starting training on Reactant GPU backend…")

    for epoch in 1:nepochs
        epoch_loss = 0.0f0
        nbatches = 0

        for (x, y) in train_loader_dev
            _, l, _, ts = Lux.Training.single_train_step!(Lux.AutoEnzyme(), loss_layer, (x, y), ts; sync = false)
            epoch_loss += l
            nbatches += 1
        end

        avg_loss = epoch_loss / max(nbatches, 1)

        ps_cpu = ts.parameters |> ProbAbEx.cpu_device()
        st_cpu = ts.states     |> ProbAbEx.cpu_device()

        train_acc = accuracy(model, ps_cpu, st_cpu, train_loader_cpu)
        test_acc  = accuracy(model, ps_cpu, st_cpu, test_loader_cpu)

        println("[$epoch/$nepochs] loss = $(round(avg_loss, digits=4)), " *
                "train acc = $(round(train_acc, digits=4)), " *
                "test acc = $(round(test_acc, digits=4))")
        # println("[$epoch/$nepochs] loss = $(round(avg_loss, digits=4))")
    end

    return ts
end
