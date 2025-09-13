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

        time = @elapsed steps, solution_subsets = forward_search(
            sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, num_samples),
            ShapleyHeuristic(sm, sampler, num_samples),
            time_limit=55,
            terminate_on_first_solution=true,
            refine_with_backward=true
        )
        CUDA.synchronize()

        if solution_subsets !== nothing && time <= 55
            subset_size = length(solution_subsets) > 0 ? length(solution_subsets) : 0
            push!(results, (image=img_i, ϵ=ϵ, num_samples=num_samples, time=time, steps=steps, subset_size=subset_size, solution=solution_subsets))
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

        time = @elapsed steps, solution_subsets = beam_search(sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, num_samples), ShapleyHeuristic(sm, sampler, num_samples); time_limit=300, beam_size=5, terminate_on_first_solution=true)

        CUDA.synchronize()

        if solution_subsets !== nothing
            solution_subsets = solution_subsets[1][3]
            subset_size = length(solution_subsets) > 0 ? length(solution_subsets) : 0
            push!(results, (image=img_i, ϵ=ϵ, num_samples=num_samples, time=time, steps=steps, subset_size=subset_size, solution=solution_subsets))
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

        xₛ, yₛ, sm = get_image(2)
        II = init_full_sbitset(xₛ)
        println("Image: $img_i, successful_images: $successful_images, ϵ: $ϵ, num_samples: $num_samples")

        time = @elapsed steps, solution_subsets = backward_search(
            sm, II, ii -> isvalid_sdp(ii, sm, ϵ, sampler, num_samples),
            ShapleyHeuristic(sm, sampler, num_samples),
            time_limit=360,
            terminate_on_first_solution=false,
        )
        
        CUDA.synchronize()

        if solution_subsets !== nothing
            subset_size = length(solution_subsets) > 0 ? length(solution_subsets) : 0
            push!(results, (image=img_i, ϵ=ϵ, num_samples=num_samples, time=time, steps=steps, subset_size=subset_size, solution=solution_subsets))
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
    push!(results, run_experiment_beam(50, 0.9, 1000))
    println("Finished 1")
    push!(results, run_experiment_beam(50, 0.99, 1000))
    println("Finished 2")
    push!(results, run_experiment_beam(50, 0.9, 10000))
    println("Finished 3")
    push!(results, run_experiment_beam(50, 0.99, 10000))
    results
end

function all_backward_results()
    results = []
    push!(results, run_experiment_backward(50, 0.9, 1000))
    println("Finished 1")
    push!(results, run_experiment_backward(50, 0.99, 1000))
    println("Finished 2")
    push!(results, run_experiment_backward(50, 0.9, 10000))
    println("Finished 3")
    push!(results, run_experiment_backward(50, 0.99, 10000))
    results
end

function all_forward_results()
    results = []
    push!(results, run_experiment_forward(100, 0.9, 1000))
    println("Finished 1")
    push!(results, run_experiment_forward(100, 0.99, 1000))
    println("Finished 2")
    push!(results, run_experiment_forward(100, 0.9, 10000))
    println("Finished 3")
    push!(results, run_experiment_forward(100, 0.99, 10000))
    results
end


all_results = []
sampler = UniformDistribution()
xₛ = train_X_bin_neg[:, 1] |> to_gpu
yₛ = argmax(model(xₛ))
sm = Subset_minimal(to_gpu(model), xₛ, yₛ)
II = init_sbitset(length(xₛ))
# II = init_full_sbitset(xₛ)


# all_results = all_forward_results()
# all_results = all_beam_results()
# all_results = all_backward_results()

# time = @elapsed steps, solution_subsets = beam_search(sm, II, ii -> isvalid_sdp(ii, sm, 0.3, sampler, 1000), ShapleyHeuristic(sm, sampler, 1000); beam_size=5, terminate_on_first_solution=true)
# time = @elapsed steps, solution_subsets = backward_search(sm, II, ii -> isvalid_sdp(ii, sm, 0.9, sampler, 1000), ShapleyHeuristic(sm, sampler, 1000))
time = @elapsed steps, solution_subsets = forward_search(sm, II, ii -> isvalid_sdp(ii, sm, 0.9, sampler, 1000), ShapleyHeuristic(sm, sampler, 1000); terminate_on_first_solution=true)
