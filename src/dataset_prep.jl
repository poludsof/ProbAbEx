
function preprocess_binary(X, threshold=0.5)
    binary_X = Float32.(X) .>= threshold
    return reshape(binary_X, 28 * 28, size(binary_X, 3))
end

function preprocess_bin_neg(data)
    bin_data_neg = 2 .* data .- 1
    return bin_data_neg
end

function onehot_labels(y)
    indicatormat(y)
end

