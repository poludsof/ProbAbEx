

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

function prepare_data()
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtest, ytest = MLDatasets.MNIST(:test)[:]
    xtrain = map(x -> 2*(x > 0.5) - 1, reshape(xtrain, :, size(xtrain,3)))
    xtest = map(x -> 2*(x > 0.5) - 1, reshape(xtest, :, size(xtest,3)))
    ytrain = ytrain .+ 1
    ytest = ytest .+ 1
    ((xtrain, ytrain), (xtest, ytest))
end

function prepare_data_only_train(seed = 1)
    xtrain, ytrain = MLDatasets.MNIST(:train)[:]
    xtrain = map(x -> 2*(x > 0.5) - 1, reshape(xtrain, :, size(xtrain,3)))
    ytrain = ytrain .+ 1
    tst_ii = sample(1:length(ytrain), 10_000, replace = false)
    trn_ii = setdiff(1:length(ytrain), tst_ii)
    xtest, ytest = xtrain[:, tst_ii], ytrain[tst_ii]
    xtrain, ytrain = xtrain[:, trn_ii], ytrain[trn_ii]
    ((xtrain, ytrain), (xtest, ytest))
end
