function  get_image(img_i)
    xₛ = train_X_bin_neg[:, img_i] |> to_gpu
    yₛ = argmax(model(xₛ))
    sm = Subset_minimal(to_gpu(model), xₛ, yₛ)
    xₛ, yₛ, sm
end

function run_experiment_forward(num_img, ϵ, num_samples)
    results = []
    img_i = 1
    successful_images = 0

    while successful_images < num_img

        xₛ, yₛ, sm = get_image(img_i)
        II = init_sbitset(length(xₛ))
        println("Image: $img_i, successful_images: $successful_images, ϵ: $ϵ, num_samples: $num_samples")

        time = @elapsed solution_subsets = forward_search(
            sm, II, ii -> isvalid_ep(ii, sm, ϵ, sampler, num_samples),
            ShapleyHeuristic(sm, sampler, num_samples),
            time_limit=100,
            terminate_on_first_solution=true,
            refine_with_backward=true
        )
        CUDA.synchronize()

        if solution_subsets !== nothing && time <= 100
            subset_size = length(solution_subsets) > 0 ? length(solution_subsets) : 0
            push!(results, (image=img_i, ϵ=ϵ, num_samples=num_samples, time=time, subset_size=subset_size, solution=solution_subsets))
            successful_images += 1
        end

        CUDA.reclaim()
        img_i += 1
    end

    return results
end

function run_experiment_beam(num_img, ϵ, num_samples)
    results = []
    img_i = 1
    successful_images = 0

    while successful_images < num_img

        xₛ, yₛ, sm = get_image(img_i)
        II = init_sbitset(length(xₛ))
        println("Image: $img_i, successful_images: $successful_images, ϵ: $ϵ, num_samples: $num_samples")

        time = @elapsed solution_subsets = beam_search(
            sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, num_samples),
            ShapleyHeuristic(sm, sampler, num_samples),
            time_limit=300, 
            beam_size=5, 
            terminate_on_first_solution=true)

        CUDA.synchronize()

        if solution_subsets !== nothing
            solution_list = collect(solution_subsets)
            best_subset = solution_list[1]
            subset_size = length(best_subset) > 0 ? length(best_subset) : 0
            push!(results, (image=img_i, ϵ=ϵ, num_samples=num_samples, time=time, subset_size=subset_size, solution=best_subset))
            successful_images += 1
        end

        CUDA.reclaim()
        img_i += 1
    end

    return results
end

function run_experiment_backward(num_img, ϵ, num_samples)
    results = []
    img_i = 1
    successful_images = 0

    while successful_images < num_img

        xₛ, yₛ, sm = get_image(img_i)
        II = init_full_sbitset(xₛ)
        println("Image: $img_i, successful_images: $successful_images, ϵ: $ϵ, num_samples: $num_samples")

        time = @elapsed solution_subsets = backward_search(
            sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, num_samples),
            ShapleyHeuristic(sm, sampler, num_samples);
            time_limit=30,
            terminate_on_first_solution=false,
        )
        
        CUDA.synchronize()

        if solution_subsets !== nothing && length(solution_subsets) > 0
            solution_list = collect(solution_subsets)
            best_subset = solution_list[1]
            subset_size = length(best_subset) > 0 ? length(best_subset) : 0
            push!(results, (image=img_i, ϵ=ϵ, num_samples=num_samples, time=time, subset_size=subset_size, solution=best_subset))
            successful_images += 1
        end

        CUDA.reclaim()
        img_i += 1
    end
    return results
end

function init_full_sbitset(xₛ)
    II = SBitSet{13, UInt64}(collect(1:length(xₛ)))
    II
end

function all_beam_results()
    results = []
    push!(results, run_experiment_beam(1, 0.9, 1000))
    println("Finished 1")
    # push!(results, run_experiment_beam(50, 0.99, 1000))
    # println("Finished 2")
    # push!(results, run_experiment_beam(50, 0.9, 10000))
    # println("Finished 3")
    # push!(results, run_experiment_beam(50, 0.99, 10000))
    # results
end

function all_backward_results()
    results = []
    push!(results, run_experiment_backward(1, 0.9, 1000))
    println("Finished 1")
    # push!(results, run_experiment_backward(50, 0.99, 1000))
    # println("Finished 2")
    # push!(results, run_experiment_backward(50, 0.9, 10000))
    # println("Finished 3")
    # push!(results, run_experiment_backward(50, 0.99, 10000))
    # results
end

function all_forward_results()
    results = []
    # push!(results, run_experiment_forward(1, 0.9, 1000))
    # println("Finished 1")
    # push!(results, run_experiment_forward(30, 0.99, 1000))
    # println("Finished 2")
    # push!(results, run_experiment_forward(30, 0.9, 10000))
    # println("Finished 3")
    # push!(results, run_experiment_forward(30, 0.99, 10000))
    results
end

using ProbAbEx
all_results = []
# sampler = UniformDistribution()
sampler_path = "models/milan_centers.jls"
sampler = BernoulliMixture(to_gpu(deserialize(sampler_path)))
xₛ = train_X_bin_neg[:, 1] |> to_gpu
yₛ = argmax(model(xₛ))
sm = ProbAbEx.Subset_minimal(model, xₛ, yₛ)
II = init_sbitset(length(xₛ))
II = init_full_sbitset(xₛ)

# push!(all_results, run_experiment_forward(20, 0.99, 10000))

# all_results = all_forward_results()
# all_results = all_beam_results()
# all_results = all_backward_results()

solution_subsets = ProbAbEx.beam_search(sm, II, ii -> ProbAbEx.isvalid_sdp(ii, sm, 0.3, sampler, 1000), ProbAbEx.ShapleyHeuristic(sm, sampler, 1000); beam_size=5, terminate_on_first_solution=true)
# time = @elapsed steps, solution_subsets = backward_search(sm, II, ii -> isvalid_sdp(ii, sm, 0.9, sampler, 1000), ShapleyHeuristic(sm, sampler, 1000))
# time = @elapsed steps, solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, 0.9, sampler, 1000), ShapleyHeuristic(sm, sampler, 1000); terminate_on_first_solution=true)


#!TEST distribution
# sampler_path = "Subset_minimal_search/models/milan_centers.jls"
# sampler = BernoulliMixture(to_gpu(deserialize(sampler_path)))
# time = @elapsed steps, solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, 0.9, sampler, 1000), ShapleyHeuristic(sm, sampler, 1000); terminate_on_first_solution=true, refine_with_backward=true)

#!TEST SDP/EP

# CUDA.synchronize()
# CUDA.reclaim()


# flat_results = vcat(all_results...)

# using DataFrames
# using CSV

# df = DataFrame(
#            image = [r.image for r in flat_results],
#            ϵ = [r.ϵ for r in flat_results],
#            num_samples = [r.num_samples for r in flat_results],
#            time = [r.time for r in flat_results],
#            steps = [r.steps for r in flat_results],
#            subset_size = [r.subset_size for r in flat_results],
#            solution = [join(r.solution, ",") for r in flat_results]
# )

# CSV.write("ep_tuned_forward_search_uniform_099_10000", df)
# df = CSV.read("/home/sofia/julia_project/ep_tuned_forward_search_uniform_all", DataFrame)

# condition11 = (df.ϵ .== 0.9) .& (df.num_samples .== 1000)
# condition12 = (df.ϵ .== 0.99) .& (df.num_samples .== 1000)
# condition21 = (df.ϵ .== 0.9) .& (df.num_samples .== 10000)
# condition22 = (df.ϵ .== 0.99) .& (df.num_samples .== 10000)
# condition11 = (data.epsilon .== 0.9) .& (data.num_samples .== 1000)
# condition12 = (data.epsilon .== 0.99) .& (data.num_samples .== 1000)
# condition21 = (data.epsilon .== 0.9) .& (data.num_samples .== 10000)
# condition22 = (data.epsilon .== 0.99) .& (data.num_samples .== 10000)

# data_condition11 = df[condition11, :]
# data_condition12 = df[condition12, :]
# data_condition21 = df[condition21, :]
# data_condition22 = df[condition22, :]


# avg_time_11 = mean(data_condition11.time)
# avg_subset_size_11 = mean(data_condition11.subset_size)

# avg_time_12 = mean(data_condition12.time)
# avg_subset_size_12 = mean(data_condition12.subset_size)

# avg_time_21 = mean(data_condition21.time)
# avg_subset_size_21 = mean(data_condition21.subset_size)

# avg_time_22 = mean(data_condition22.time)
# avg_subset_size_22 = mean(data_condition22.subset_size)

# conditions = [
#     (0.99, 1000),  # ϵ=0.9, n=1000
#     (0.99, 10000)  # ϵ=0.9, n=10000
# ]
# table_data = Array{Any}(missing, 2, 4)  # 3 rows (forward, backward, beam), 4 columns

# for (col_idx, (ϵ_val, n_val)) in enumerate(conditions)
#     subset = filter(row -> row.ϵ == ϵ_val && row.num_samples == n_val, df)
#     if nrow(subset) > 0
#         avg_subset_size = round(mean(subset.subset_size), digits=2)
#         table_data[3, col_idx] = "$avg_subset_size"
#     else
#         table_data[3, col_idx] = "-"
#     end
# end

# for row in 1:2
#     for col in 1:4
#         table_data[row, col] = "-"
#     end
# end

# row_names = ["Forward Search", "Backward Search", "Beam Search"]
# col_names = ["", "ϵ=0.8 & n=1000", "ϵ=0.9 & n=1000", "ϵ=0.8 & n=10000", "ϵ=0.9 & n=10000"]

# latex_code = "\\begin{tabular}{|c|c|c|c|c|}\n"
# latex_code *= "\\hline\n"

# latex_code *= join(col_names, " & ")
# latex_code *= " \\\\\n"
# latex_code *= "\\hline\n"

# for i in 1:3
#     row = [row_names[i]; table_data[i, :]]
#     latex_code *= join(row, " & ")
#     latex_code *= " \\\\\n"
# end

# latex_code *= "\\hline\n"
# latex_code *= "\\end{tabular}\n"

# open("subset_size_table.tex", "w") do io
#     write(io, latex_code)
# end
