using Serialization, Dates, TimerOutputs

struct ExperimentResult
    image_idx::Int
    threshold::Float64
    solution::Any          # the SBitSet subset (or nothing if not found)
    subset_size::Int
    time_seconds::Float64
    found::Bool
end

function run_experiments(train_X_binary, train_y, nn, compiled_acc_looped, model_cls, ps_dev, st_dev, sampler, dev; n_images=10, thresholds=[0.9, 0.99], output_dir="results")
    mkpath(output_dir)
    results = ExperimentResult[]

    success_img_count = 0
    img_idx = 2
    while success_img_count < n_images*2 && img_idx <= size(train_X_binary, 2)
        println("\n========== Image $img_idx / $n_images ==========")

        # set up sm for this image
        xₛ = Float32.(train_X_binary[:, img_idx])
        yₛ = argmax(train_y[:, img_idx])
        sm = PAE.Subset_minimal(nn, xₛ, yₛ)
        II = init_sbitset(length(xₛ), 1)

        for ϵ in thresholds
            println("\n--- Image $img_idx, threshold = $ϵ ---")

            to_exp = TimerOutput()
            solution = nothing
            found = false
            
            start_time = time()
            @timeit to_exp "forward_search" begin
                solution = PAE.forward_search(
                    sm, II,
                    ii -> PAE.isvalid_sdp_batched(ii, sm, ϵ, sampler, compiled_acc_looped, model_cls, ps_dev, st_dev),
                    PAE.ShapleyHeuristic(sm, sampler, 0.99, compiled_acc_looped, model_cls, ps_dev, st_dev);
                    time_limit=7000.0,
                    refine_with_backward=false,
                    terminate_on_first_solution=true
                )
                elapsed = time() - start_time
                found = solution !== nothing && length(solution) > 0 && elapsed <= 10000.0
            end

            elapsed = TimerOutputs.time(to_exp["forward_search"]) / 1e9  # ns → seconds
            sz = found ? length(solution) : 0

            println("Found: $found | Size: $sz | Time: $(round(elapsed, digits=2))s")

            if found
                push!(results, ExperimentResult(img_idx, ϵ, solution, sz, elapsed, found))
                success_img_count += 1
            end
        end
        img_idx += 1
    end

    # save full results (with SBitSets) for later reloading
    jls_path = joinpath(output_dir, "results_3.jls")
    serialize(jls_path, results)
    println("\nSaved serialized results to $jls_path")

    # save human-readable CSV
    csv_path = joinpath(output_dir, "results_3.csv")
    open(csv_path, "w") do f
        println(f, "image_idx,threshold,found,subset_size,time_seconds")
        for r in results
            println(f, "$(r.image_idx),$(r.threshold),$(r.found),$(r.subset_size),$(round(r.time_seconds, digits=4))")
        end
    end
    println("Saved CSV to $csv_path")

    return results
end

results = run_experiments(
    train_X_binary, train_y,
    nn, compiled_acc_looped,
    model_cls, ps_dev, st_dev,
    sampler, dev;
    n_images = 2,
    thresholds=[0.9, 0.99],
    output_dir="results"
)