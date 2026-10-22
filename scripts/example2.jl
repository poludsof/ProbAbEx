
"""
Beam search with b=1 with partial and full probability criterion
"""

using Subset_minimal_search

using Subset_minimal_search.Flux
using Subset_minimal_search.MLDatasets
using Subset_minimal_search.StaticBitSets
using Subset_minimal_search.Makie
using Subset_minimal_search.CairoMakie
using Subset_minimal_search.Serialization

using Subset_minimal_search: preprocess_binary, preprocess_bin_neg, onehot_labels, accuracy, Subset_minimal, full_beam_search


# Load model
model_path = joinpath(@__DIR__, "..", "models", "binary_model.jls")
model = deserialize(model_path)


# Prepare data
train_X, train_y = MNIST(split=:train)[:]
test_X, test_y = MNIST(split=:test)[:]

train_X_binary = preprocess_binary(train_X)
test_X_binary = preprocess_binary(test_X)

train_X_bin_neg = preprocess_bin_neg(train_X_binary)
test_X_bin_neg = preprocess_bin_neg(test_X_binary)

train_y = onehot_labels(train_y)
test_y = onehot_labels(test_y)
 

# Prepare image and label
img = train_X_bin_neg[:, 1]
label_img = argmax(train_y[:, 1]) - 1


# Greedy approach + backward search
threshold = 0.1
num_samples = 1000
best_set = greedy_subsets_search(Subset_minimal(model, img, label_img), threshold, num_samples)
reduced_sets = backward_priority_reduction(Subset_minimal(model, img, label_img), best_set, threshold, num_samples)
#TEST reduced sets
max_error(model, img, best_set, 0.9, num_samples)
max_error(model, img, reduced_sets, 0.9, num_samples)


# Forward search with priority stack + backward search
threshold = 0.1
num_samples = 1000
fwd_best_set = forward_priority_search(Subset_minimal(model, img, label_img), threshold, num_samples)
fwd_reduced_sets = backward_priority_reduction(Subset_minimal(model, img, label_img), fwd_best_set, threshold, num_samples)
# Test reduced sets
max_error(model, img, fwd_best_set, 0.9, num_samples)
max_error(model, img, fwd_reduced_sets, 0.9, num_samples)


# Search for subsets implicating indices of 2nd and 3rd layers
I3, I2, I1 = fwd_reduced_sets
subset_threshold = 0.9
num_samples = 1500
subsubset_I2 = implicative_subsets(model[1], img, I3, I2, subset_threshold, num_samples)
subsubset_I1 = implicative_subsets(model[2], model[1](img), I2, I1, subset_threshold, num_samples)
#check implicative_subsets
# sdp_partial(model[1], img, I3, I2, num_samples)
# max_error(model, img, (I3, I2, I1), 0.9, num_samples)


"""RESULTS:"""
# Length of (I3, I2, I1): (71, 25, 31)
# Reduced length: (54, 21, 26) 

# Reduced (I3, I2, I1):
#=
fwd_reduced_sets = [SBitSet{13,UInt64}([7,63,73,79,96,101,110,133,145,155,161,163,174,191,201,207,232,237,242,257,296,298,299,300,301,302,326,328,330,338,355,357,358,359,360,378,390,404,417,432,441,489,538,559,567,602,604,626,636,643,647,700,705,729]),
                    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
                    SBitSet{4,UInt64}([5,7,8,9,12,19,23,28,32,48,51,52,76,113,151,154,204,206,208,215,216,235,246,248,252,255])]
I3, I2, I1 = fwd_reduced_sets
=#

# Implicative subsets for I2 and I1:
#=
subsubset_I2 = [
    SBitSet{13,UInt64}([7,96,101,110,155,161,174,201,237,242,257,298,301,326,328,330,338,378,432,559,602,604,626,729]),
    SBitSet{13,UInt64}([110,133,145,161,163,174,242,257,296,299,301,302,378,404,417,538,559,636]),
    SBitSet{13,UInt64}([79,96,101,145,155,161,174,201,207,232,301,302,326,338,359,432,441,489,559,567,647,700,705,729]),
    SBitSet{13,UInt64}([7,73,79,96,145,155,161,174,300,302,330,360,626,700,705]),
    SBitSet{13,UInt64}([73,79,174,302,330,355,357,359,360,489,626,705,729]),
    SBitSet{13,UInt64}([7,63,101,155,201,257,300,326,328,338,357,390,441,538,559,567,705]),
    SBitSet{13,UInt64}([7,63,101,110,145,155,163,237,296,298,299,301,338,538,567,602,643,705,729]),
    SBitSet{13,UInt64}([7,63,79,101,110,133,155,174,191,207,237,242,298,300,326,328,330,359,360,432,559,602,604,626,647,705,729]),
    SBitSet{13,UInt64}([73,101,110,133,145,161,163,191,232,242,298,300,302,328,338,357,359,360,404,417,432,538,626,647,729]),
    SBitSet{13,UInt64}([63,73,79,133,145,155,161,174,191,201,232,242,296,298,299,301,302,330,355,357,358,390,432,441,489,559,626,700]),
    SBitSet{13,UInt64}([63,79,133,161,174,201,232,242,296,298,301,328,355,358,359,360,441,489,559,602,636]),
    SBitSet{13,UInt64}([79,101,110,155,207,237,296,299,300,301,302,326,330,355,359,390,432,538,559,602,604,643,705]),
    SBitSet{13,UInt64}([7,63,73,79,96,101,110,133,145,155,161,163,174,191,201,207,232,237,242,257,296,298,299,300,301,302,326,328,330,338,355,357,358,359,360,378,390,404,417,432,441,489,538,559,567,602,604,626,636,643,647,700,705,729]),
    SBitSet{13,UInt64}([96,133,145,163,191,207,232,237,257,298,390,441,602]),
    SBitSet{13,UInt64}([96,110,133,242,298,300,301,328,355,359,378,404,417,432,538,559,567,602,636,643,705,729]),
    SBitSet{13,UInt64}([79,96,161,174,191,207,242,257,296,338,355,358,360,404,417,559,567,602,636]),
    SBitSet{13,UInt64}([73,96,101,155,161,191,232,237,257,298,300,302,326,328,338,357,359,360,378,390,441,602,604,636,643,647,700]),
    SBitSet{13,UInt64}([7,63,73,79,96,101,110,133,145,155,161,163,174,191,201,207,232,237,242,257,296,298,299,300,301,302,326,328,330,338,355,357,358,359,360,378,390,404,417,432,441,489,538,559,567,602,604,626,636,643,647,700,705,729]),
    SBitSet{13,UInt64}([63,73,79,101,110,133,155,161,163,174,207,237,296,302,338,357,359,378,390,602,626,636,643,700,729]),
    SBitSet{13,UInt64}([73,96,101,110,133,155,161,163,201,207,301,326,328,330,358,360,390,404,432,538,602,626,636,643,729]),
    SBitSet{13,UInt64}([7,63,73,79,96,110,145,155,174,191,207,232,242,302,328,330,338,357,358,359,360,390,417,432,489,559,647,700])
]
subsubset_I1 = [
    SBitSet{4,UInt64}([11,42,83,85,156,183,190,209,224]),
    SBitSet{4,UInt64}([42,85,112,183,190]),
    SBitSet{4,UInt64}([27,42,186,190,197]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([27,67,157,183,186,190,197,217]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([11,60,69,85,209]),
    SBitSet{4,UInt64}([27,42,60,84,85,156,183,209,217]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([11,32,42,60,84,112,156,157,191,197,217]),
    SBitSet{4,UInt64}([11,67,83,84,112,156,186,191,209]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([42,60,69,83,156,157,191,224]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([32,42,60,67,69,83,85,156,157,186,190,191,209,217,224]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([27,84,85,183,217]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]),
    SBitSet{4,UInt64}([32,60,67,69,83,156,157]),
    SBitSet{4,UInt64}([27,156,186,209,224]),
    SBitSet{4,UInt64}([42,84,85,157,209,217]),
    SBitSet{4,UInt64}([11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224])
]
=#