
# I3 = [7,63,73,79,96,101,110,133,145,155,161,163,174,191,201,207,232,237,242,257,296,298,299,300,301,302,326,328,330,338,355,357,358,359,360,378,390,404,417,432,441,489,538,559,567,602,604,626,636,643,647,700,705,729]
# I2 = [11,27,32,42,60,67,69,83,84,85,112,156,157,183,186,190,191,197,209,217,224]
# I1 = [5,7,8,9,12,19,23,28,32,48,51,52,76,113,151,154,204,206,208,215,216,235,246,248,252,255]

# I3, I2, I1 = stets_to_vis
# I2 = [i for i in I2]
# fig = Figure()
fig = Figure(resolution = (2000, 1800))
# Number of neurons per layer
neurons_per_column = [1, length(I2), length(I1), 1]

# A grid for each layer (arranged in a column)
col_grids = [GridLayout(fig[1, i]; valign = :center) for i in eachindex(neurons_per_column)]
# An axis per neuron
col_axes = [
    [Axis(col_grids[i][j, 1]) for j in 1:neurons_per_column[i]]
    for i in eachindex(neurons_per_column)
]

function compute_indices(col_idx, row_idx, I3, subsubset_I1, subsubset_I2, I2)
    if col_idx == 1
        return collect(I3)
    elseif col_idx == 2
        return collect(subsubset_I2[row_idx])
    elseif col_idx == 3
        indices = []
        for neuron in subsubset_I1[row_idx]
            match_index = findfirst(x -> x == neuron, I2)
            if !isnothing(match_index)
                indices = union(indices, collect(subsubset_I2[match_index]))
            end
        end
        return indices
    else
        return collect(I3)
    end
end

# Plot into the axes
for (col_idx, col_of_axes) in enumerate(col_axes)
    for (row_idx, axis) in enumerate(col_of_axes)
        img = train_X_bin_neg[:, 1]
        hidedecorations!(axis)
        hidespines!(axis)
        axis.aspect[] = DataAspect()
        axis.yreversed[] = true

        ii = compute_indices(col_idx, row_idx, I3, subsubset_I1, subsubset_I2, I2)
        img[ii] .= 2
        img_reshaped = reshape(img, 28, 28)
        heatmap!(axis, img_reshaped; colormap = :viridis)
    end
end

# Size the columns that are not the input/output columns appropriately
for (nlayers, grid) in collect(zip(neurons_per_column, col_grids))[begin+1:end-1]
    grid.tellheight[] = false
    grid.height[] = Relative(nlayers / maximum(neurons_per_column))
end

# Connect neurons
# First, figure out what neurons to connect
important_neurons = [[] for n in neurons_per_column]
# important_neurons = Vector{Vector{Int}}[[], [], [], []]
important_neurons[1] = [[1]]
important_neurons[2] = [[1] for i in collect(1:length(I2))]
important_neurons[3] = [[] for i in collect(1:length(I1))]
important_neurons[4] = [[1:length(I1)...]]

for (i, subset) in enumerate(subsubset_I1)
    indices = [
                findfirst(x -> x == neuron, I2) 
                for neuron in subset if !isnothing(findfirst(x -> x == neuron, I2))
                ]
    append!(important_neurons[3][i], indices)
end

# This utility function computes the "east" and "west" anchors (in latex parlance)
function _compute_left_right_anchors(bbox_obs)
    left_anchor = @lift(Point2($(bbox_obs).origin[1], $(bbox_obs).origin[2] + $(bbox_obs).widths[2] / 2))
    right_anchor = @lift(Point2($(bbox_obs).origin[1] + $(bbox_obs).widths[1], $(bbox_obs).origin[2] + $(bbox_obs).widths[2] / 2))
    # return NamedTuple of observables
    return (; left = left_anchor, right = right_anchor)
end

# Compute anchors-as-observables for each axis
axis_anchors = [
    [
        _compute_left_right_anchors(axis.scene.viewport)
        for axis in col_of_axes
    ]
    for col_of_axes in col_axes
]
# Connect them up!
for layer_idx in collect(eachindex(neurons_per_column))[2:end]
    neurons_for_this_layer = important_neurons[layer_idx]
    println("layer: ", layer_idx)
    for dest_idx in collect(1:neurons_per_column[layer_idx])
        println("neuron: $dest_idx, ", "its subset for prev layer: ", neurons_for_this_layer[dest_idx])
        for source_idx in neurons_for_this_layer[dest_idx]
            println("source: ", source_idx)
            lines!(
                fig.scene, 
                lift(axis_anchors[layer_idx-1][source_idx].right, axis_anchors[layer_idx][dest_idx].left) do start, stop
                    rel = stop - start
                    c1 = Point2(start[1] + rel[1]/4, start[2])
                    c2 = Point2(stop[1] - rel[1]/4, stop[2])
                    return BezierPath([MoveTo(start), CurveTo(c1, c2, stop)])
                end;
                color = :black,
            )
        end
    end
end

fig

save("my_figure_test.png", fig)


################################################################
# subsubset_I1 = [[10, 2], [8], [4, 5]]
# important_neurons[3] = [[] for i in collect(1:length(subsubset_I1))]
# I2 = [2, 4, 5, 8, 10]
# for (i, subset) in enumerate(subsubset_I1)
#     for neuron in subset
#         index = findfirst(x -> x == neuron, I2)
#         if !isnothing(index)
#             # println("Found for $neuron, index is $index in I2")
#             push!(important_neurons[3][i], index)
#         end
#     end
# end
# println(important_neurons[3])

# for (i, subset) in enumerate(subsubset_I1)
#     indices = [
#                 findfirst(x -> x == neuron, I2) 
#                 for neuron in subset if !isnothing(findfirst(x -> x == neuron, I2))
#                 ]
#     append!(important_neurons[3][i], indices)
# end
