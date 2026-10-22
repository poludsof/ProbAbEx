# srun -p gpufast --gres=gpu:1  --mem=16000  --pty bash -i

# cd julia/Pkg/ProbAbEx/tests/
#  ~/.juliaup/bin/julia --project=.
using CUDA
using ProbAbEx
import ProbAbEx as PAE
using ProbAbEx.Flux
using ProbAbEx.LinearAlgebra
using ProbAbEx.MLDatasets
using ProbAbEx.StaticBitSets
using ProbAbEx.TimerOutputs
using Serialization
using Distributions
using ProbAbEx.Makie
using ProbAbEx: Subset_minimal
const to = ProbAbEx.to
const PAE = ProbAbEx

# include("models.jl")
# include("rule_utilities.jl")
# include("beam_search.jl")
# include("heuristics_criteria.jl")
# include("training_utilities.jl")

empty_sbitset(n::Int) = SBitSet{ceil(Int, n / 64), UInt64}()
empty_sbitset(x::AbstractArray) = empty_sbitset(length(x))

function sbitset(n::Int, ii) 
    rule = SBitSet{ceil(Int, n / 64), UInt64}()
    foldl((r,i) -> push(r, i), ii, init = rule)
end

full_sbitset(n::Int) = sbitset(n, 1:n)
full_sbitset(x::AbstractArray) = full_sbitset(length(x))

to_gpu = gpu
# to_gpu = cpu
(train_x, train_y), (test_x, test_y) = prepare_data() |> to_gpu;
cpu_train_y = cpu(train_y)

# Let's get only correct IDs
# models = (("model_l2", model_l2()), ("model_l1", model_l1()), ("model_noise", model_noise()), ("model_default", model_default()))
# mask = mapreduce((x,y) -> x .& y, models) do (_, model )
#     model = gpu(model)
#     Flux.onecold(model(train_x)) .== train_y
# end |> cpu
# correct_img_ids = mapreduce(vcat, 1:10) do c
#     ii = findall(cpu_train_y .== c)
#     ii = filter(i -> mask[i], ii)
#     sort(ii)[1:10]
# end

#? sampler = UniformDistribution()
#? sampler = BernoulliMixture(to_gpu(deserialize(joinpath("models", "milan_centers.jls"))))
#? ϵ = 0.99
sampler_path = "Subset_minimal_search/models/milan_centers.jls"
sampler = BernoulliMixture(to_gpu(deserialize(sampler_path)))
################################################################################
# Let's generate samples for fitting the di
################################################################################
# model = deserialize("mnist/binary_model.jls") |> to_gpu;

# serialize("mnist/binary_model_hidden.jls", cpu(vcat(train_x, model[1](train_x), model[1:2](train_x))))

################################################################################
# Let's now try to find the "skeleton of a network"
################################################################################
#? model = deserialize("/home/sofia/julia_project/Subset_minimal_search/ml_in_prague/mnist/binary_model.jls") |> to_gpu;
 centers = to_gpu(deserialize("/home/sofia/julia_project/Subset_minimal_search/ml_in_prague/mnist/binary_model_hidden_centers.jls"))
#? joint_sampler = BernoulliMixture(centers)
#? input_sampler = BernoulliMixture(centers[:,1:784,:])
 samplers = (BernoulliMixture(centers[:,1:784,:]), BernoulliMixture(centers[:,785:1040,:]), BernoulliMixture(centers[:,1041:end,:]))


function finalize(hs)
    hs₊ = map(x -> max(0, ϵ - x), hs) # the volume I am missing to meet the threshold
    h₊ = mean(hs₊) + maximum(hs₊)
    h₊ > 0 && return(h₊)
    hs₋ = map(x -> max(0, x - ϵ), hs) # the volume I am missing to meet the threshold
    h₋ = mean(hs₋) + maximum(hs₋)
    -h₋
end 

ϵ = 0.9
img_id = 2

#! ################################################################################
#= DELETE MY OLD CODE FOR BC
cc = cpu(Flux.onecold(model(train_x)) .== train_y)
img_ids = [findfirst(i -> cpu_train_y[i] == j && cc[i], eachindex(cc)) for j in 1:1]
for img_id in img_ids
    xₛ = train_x[:, img_id] |> to_gpu
    yₛ = model(xₛ)
    sm = Subset_minimal(model, xₛ, yₛ, (784, 256, 256))
    ii = (empty_sbitset(784), empty_sbitset(256), empty_sbitset(256))

    println("yₛ = ")
    heuristic = BatchHeuristic(
        ii -> accuracy_sdp3(ii, sm, samplers, 10000; verbose = true),
        ii -> batch_heuristic3(ii, sm, samplers, 100000; verbose = true),
        finalize
        )
    rule = forward_search(sm, ii, ii -> isvalid_sdp3(ii, sm, ϵ, samplers, 100000; verbose = true), heuristic; terminate_on_first_solution = true, only_smaller = false, refine_with_backward = false)
    @show rule
    println("finished forward search = ", length(rule), " ", rule[1])
    # rule = backward_search(sm, rule, ii -> isvalid_sdp3(ii, sm, ϵ, samplers, 100000; verbose = true),  depth_first; terminate_on_first_solution = true)
    println("start skeleton")
    skeleton = map(1:length(rule)-1) do i 
        map(rule[i+1]) do j
            println("i = ", i, " j = ", j)
            xᵢ = i > 1 ? model[i-1](xₛ) : xₛ
            # sm = Subset_minimal(restrict_output(model[i], [j]), xᵢ)
            println("sm is done")
            # brule = backward_search(sm, rule[i], ii -> isvalid_sdp(ii, sm, ϵ, samplers[i], 10000),  depth_first; time_limit = 600, terminate_on_first_solution = true)
            println("I: $i", " J: $j")
            jj = findfirst(==(j), new_sets[i+1])
            println("jj = ", jj)
            if i == 1
                brule = create_brule(empty_sbitset(784), subsubset_I2[jj])
            elseif i == 2
                brule = create_brule(empty_sbitset(256), subsubset_I1[jj])
            end
            println("brule = ", brule)
            (;i, j, rule = brule)
            # brules = backward_search(sm, rule[i], ii -> isvalid_sdp(ii, sm, ϵ, input_sampler, 10000),  depth_first; time_limit = 600, terminate_on_first_solution = false)
            # (;i, j, rule = argmin(length, brules))
        end
    end
    println("finished skeleton")
    # push!(skeleton, [(;i = length(rule), j = 1, rule = rule[end])]) # let's add the last set as a rule
    # serialize("/home/sofia/julia_project/Subset_minimal_search/ml_in_prague/mnist/skeleton_100_new.jls", (;rule, skeleton, xₛ = cpu(xₛ), yₛ = cpu(yₛ)))
end
=#
#! ################################################################################

cc = cpu(Flux.onecold(model(train_x)) .== train_y)
img_ids = [findfirst(i -> cpu_train_y[i] == j && cc[i], eachindex(cc)) for j in 1:10]
for img_id in img_ids
    xₛ = train_x[:, img_id] |> to_gpu
    yₛ = model(xₛ)
    sm = Subset_minimal(model, xₛ, yₛ, (784, 256, 256))
    ii = (empty_sbitset(784), empty_sbitset(256), empty_sbitset(256))

    heuristic = BatchHeuristic(
        ii -> accuracy_sdp3(ii, sm, samplers, 1000; verbose = true),
        ii -> batch_heuristic3(ii, sm, samplers, 10000; verbose = true),
        finalize
        )
    println(typeof(heuristic))
    rules = forward_search(sm, ii, ii -> isvalid_sdp3(ii, sm, ϵ, samplers, 100000; verbose = true), heuristic; terminate_on_first_solution = true, only_smaller = false, refine_with_backward = false)
    # rules = beam_search(sm, ii, ii -> isvalid_sdp3(ii, sm, ϵ, samplers, 1000; verbose = true), heuristic; terminate_on_first_solution = true, beam_size = 5, time_limit = 300)
    # rules = forward_search(sm, ii, ii -> isvalid_sdp3(ii, sm, ϵ, samplers, 100000; verbose = true), batch_heuristic3(ii, sm, samplers, 10000), terminate_on_first_solution=true, refine_with_backward=false)
    @show rules
    rule = collect(rules)[1]
    rule = backward_search(sm, rule, ii -> isvalid_sdp3(ii, sm, ϵ, samplers, 1000; verbose = true), depth_first; terminate_on_first_solution = true)

    skeleton = map(1:length(rule)-1) do i 
        map(rule[i+1]) do j
            xᵢ = i > 1 ? model[i-1](xₛ) : xₛ
            sm = Subset_minimal(restrict_output(model[i], [j]), xᵢ)
            brule = backward_search(sm, rule[i], ii -> isvalid_sdp(ii, sm, ϵ, samplers[i], 10000),  depth_first; time_limit = 600, terminate_on_first_solution = true)
            (;i, j, rule = brule)
            # brules = backward_search(sm, rule[i], ii -> isvalid_sdp(ii, sm, ϵ, input_sampler, 10000),  depth_first; time_limit = 600, terminate_on_first_solution = false)
            # (;i, j, rule = argmin(length, brules))
        end
    end
    push!(skeleton, [(;i = length(rule), j = 1, rule = rule[end])]) # let's add the last set as a rule
    # serialize("mnist/skeleton_img=$(img_id)_$(Int(100*ϵ)).jls", (;rule, skeleton, xₛ = cpu(xₛ), yₛ = cpu(yₛ)))
end

println("finished 2")

# ################################################################################
# # The code below finds large rules number of rules from one sammple and evaluates its performance
# ################################################################################
# model = to_gpu(model_default());
# sampler = UniformDistribution()
# img_id = 2
# xₛ = train_x[:, img_id] |> to_gpu
# yₛ = Flux.onecold(model(xₛ))
# sm = Subset_minimal(model, xₛ, yₛ)

# heuristic = BatchHeuristic(
#     ii -> accuracy_sdp(ii, sm, sampler, 10000),
#     ii -> batch_heuristic(ii, sm, sampler, 100000; verbose = true),
#     x -> ϵ - x
#     )
# t = @elapsed rules = forward_search(sm, empty_sbitset(xₛ), ii -> isvalid_sdp(ii, sm, ϵ, sampler, 10000; verbose = false),  heuristic; time_limit = 3.5*3600, terminate_on_first_solution = false, only_smaller = false, refine_with_backward = false)
# serialize("mnist/model_default_uniform_forward_img_$(img_id).jls", (;rules, x = cpu(xₛ), y = yₛ))

# rules, xₛ, y = deserialize("mnist/model_default_uniform_forward_img_$(img_id).jls")
# xₛ = to_gpu(xₛ)
# yₛ = y
# stats = map(rules) do rule
#     train_recall, train_precision = rule_stats(train_x, train_y, rule, xₛ, yₛ)
#     test_recall, test_precision = rule_stats(train_x, train_y, rule, xₛ, yₛ)
#     (;rule, train_recall, train_precision, test_recall, test_precision)
# end
# serialize("mnist/model_default_uniform_forward_img_$(img_id).jls", (;stats, x = cpu(xₛ), y = yₛ))


# ################################################################################
# #   Beam-search to find normal explanations of models
# ################################################################################

# # for (model_name, model) in (("model_l2", model_l2()), ("model_l1", model_l1()), ("model_noise", model_noise()), ("model_lasso", model_lasso()), ("model_default", model_default()))
# for (model_name, model) in (("lenet_default", model_lenet()), )
#     model = gpu(model);
#     results = []
#     sname = sampler isa BernoulliMixture ? "data" : "uniform"
#     filename = "mnist/$(model_name)_$(sname)_forward_beamsearch.jls"
#     _img_ids = correct_img_ids
#     if isfile(filename)
#         results = deserialize(filename)
#         _img_ids = setdiff(correct_img_ids, unique([x.img_id for x in results]))
#     end
#     @show _img_ids
#     for img_id in _img_ids
#         @show img_id
#         xₛ = train_x[:, img_id] |> to_gpu
#         yₛ = Flux.onecold(vec(model(xₛ)))
#         sm = Subset_minimal(model, xₛ, yₛ)

#         heuristic = BatchHeuristic(
#             ii -> accuracy_sdp(ii, sm, sampler, 10000),
#             ii -> batch_heuristic(ii, sm, sampler, 10000; verbose = true),
#             x -> ϵ - x
#         )

#         # heuristic = BatchHeuristic(
#         #     ii -> accuracy_sdp(ii, sm, sampler, 10000),
#         #     ii -> batch_beta(ii, sm, sampler, 10000, ϵ; verbose = true),
#         #     x -> 1-exp(x)
#         # )

#         t = @elapsed rules = forward_beamsearch(sm, empty_sbitset(xₛ), ii -> ϵ - accuracy_sdp(ii, sm, sampler, 10000),  heuristic; time_limit = 300, terminate_on_first_solution=false, refine_with_backward = true)
#         for rule in rules
#             train_recall, train_precision = rule_stats(train_x, train_y, rule, xₛ, yₛ)
#             test_recall, test_precision = rule_stats(test_x, test_y, rule, xₛ, yₛ)
#             o = (;  
#                 x = cpu(xₛ),
#                 y = (CUDA.@allowscalar train_y[img_id]),
#                 ŷ = yₛ,
#                 train_recall = train_recall,
#                 train_precision = train_precision,
#                 test_recall = test_recall,
#                 test_precision = test_precision,
#                 time = t,
#                 img_id,
#                 rule,
#             )
#             push!(results, o)
#         end
#         serialize("mnist/$(model_name)_$(sname)_forward_beamsearch.jls", results)
#     end
# end

# ################################################################################
# #   Beam-search to find explanations of adversarial samples
# ################################################################################
# for (model_name, model) in (("model_l2", model_l2()), ("model_l1", model_l1()), ("model_noise", model_noise()), ("model_lasso", model_lasso()), ("model_default", model_default()))
#     model = gpu(model);
#     cpu_model = cpu(model)
#     results = []
#     sname = sampler isa BernoulliMixture ? "data" : "uniform"
#     filename = "mnist/$(model_name)_adversarial_$(sname)_forward_beamsearch.jls"
#     _img_ids = correct_img_ids
#     if isfile(filename)
#         results = deserialize(filename)
#         _img_ids = setdiff(correct_img_ids, unique([x.img_id for x in results]))
#     end
#     @show _img_ids

#     for img_id in _img_ids
#         @show img_id
#         xₛ = train_x[:, img_id] |> to_gpu
#         yₛ = Flux.onecold(vec(model(xₛ)))
#         xₐ = to_gpu(attack(cpu_model, cpu(xₛ)))
#         yₐ = Flux.onecold(model(xₐ))
#         sm = Subset_minimal(model, xₐ, yₐ)
#         heuristic = BatchHeuristic(
#             ii -> accuracy_sdp(ii, sm, sampler, 100000),
#             ii -> batch_heuristic(ii, sm, sampler, 10000; verbose = true),
#             x -> ϵ - x
#         )

#         t = @elapsed rules = forward_beamsearch(sm, empty_sbitset(xₐ), ii -> ϵ - isvalid_sdp(ii, sm, sampler, 10000),  heuristic; time_limit = 300, terminate_on_first_solution=false, refine_with_backward = true)
#         for rule in rules
#             train_recall, train_precision = rule_stats(train_x, train_y, rule, xₐ, yₐ)
#             test_recall, test_precision = rule_stats(test_x, test_y, rule, xₐ, yₐ)
#             o = (;  
#                 x = cpu(xₛ),
#                 xa = cpu(xₐ),
#                 y = (CUDA.@allowscalar train_y[img_id]),
#                 ŷ = yₛ,
#                 ŷa = yₐ,
#                 train_recall = train_recall,
#                 train_precision = train_precision,
#                 test_recall = test_recall,
#                 test_precision = test_precision,
#                 time = t,
#                 img_id,
#                 rule,
#             )
#             push!(results, o)
#         end
#         serialize(filename, results)
#     end
# end
