"""
Forward search with priority queue based on criterion value for minimal subset search for the FiRST layer of a neural network.
"""
function forward_search(sm, ii::TT, isvalid::Function, heuristic_fun; time_limit=Inf, terminate_on_first_solution=true, exclude_supersets = true, only_smaller = true, refine_with_backward=true, samplers=nothing, num_samples=10000) where {TT}
    println("ffffforward_search")
    initial_heuristic = heuristic_fun(ii)

    if typeof(initial_heuristic) == Tuple
        println("TUPLE H")
        initial_h, initial_hmax_err = initial_heuristic
    elseif typeof(initial_heuristic) == Number 
        initial_h = initial_heuristic
    else
        initial_h = sum(initial_heuristic)
    end
    println("Initial h: ", initial_h)

    stack = [(initial_h, ii)]
    closed_list = Set{TT}()
    solutions = Set{TT}()

    steps = 0
    start_time = time()
    smallest_solution = typemax(Int)

    @timeit to "forward search" while !isempty(stack)

        # steps > max_steps && break
        steps += 1
        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solutions")
            return solutions
        end

        sort!(stack, by = x -> -x[1])
        current_h, ii = pop!(stack)
        closed_list = push!(closed_list, ii)

        # Look only for strictly smaller solutions
        if only_smaller
            length(ii) ≥ smallest_solution && continue
        end

        # check if we have not find a solution, which is subset of solution we have find
        if exclude_supersets
            issub = @timeit to "issubset" any(Base.Fix1(issubset,ii), solutions)
            issub && println("skipping superset")
            issub && continue
        end
        
        v = @timeit to "isvalid" isvalid(ii)
        # println("isvalid: ", v)

        if v
            if refine_with_backward
                println("Valid subset found. Pruning with backward search...")
                bcwd_solutions  = backward_search(sm, ii, isvalid, heuristic_fun, time_limit=30, terminate_on_first_solution=false)
                println("After pruning: ", bcwd_solutions)
                ii = collect(bcwd_solutions)[1] # take the first solution
            end

            println("Valid subset found: $(solution_length(ii)) with heuristic: ", current_h)
            terminate_on_first_solution && return ii
            if solution_length(ii) < smallest_solution
                smallest_solution = solution_length(ii)
                println("new smallest solution so far: ", ii)
            end
            push!(solutions, ii)
            continue
        end
        println("step: $steps, length $(solution_length(ii)) with heuristic: ", current_h)

        stack = @timeit to "expand_frwd" expand_frwd(sm, stack, closed_list, ii, heuristic_fun)
    end

    println("Stack is empty")
    return solutions
end


solution_length(ii::Tuple) = solution_length.(ii)
solution_length(ii::SBitSet) = length(ii)
solution_length(::Nothing) = 0

new_subsets_fwrd(ii::SBitSet, idim) = [push(ii, i) for i in setdiff(1:idim, ii)]

function new_subsets_fwrd(ii, idims)
    if ii isa Tuple
        I1, I2, I3 = ii
    else
        # append!(new_subsets, [push(ii, i) for i in setdiff(1:idims[1], ii)])
        return [push(ii, i) for i in setdiff(1:idims[1], ii)]
    end

    new_subsets = T[]
    if I3 !== nothing
        append!(new_subsets, [(push(I3, i), I2, I1) for i in setdiff(1:idims[1], I3)])
    end
    if I2 !== nothing
        append!(new_subsets, [(I3, push(I2, i), I1) for i in setdiff(1:idims[2], I2)])
    end
    if I1 !== nothing
        append!(new_subsets, [(I3, I2, push(I1, i)) for i in setdiff(1:idims[3], I1)])
    end
    new_subsets
end

function expand_frwd(sm, stack, closed_list, ii, heuristic_fun)
    # println("Expanding subset")
    for new_subset in new_subsets_fwrd(ii, sm.dims)
        if new_subset ∉ closed_list
            new_heuristic = @timeit to "heuristic" heuristic_fun(new_subset)
            if new_heuristic isa NamedTuple
                new_heuristic, new_error = new_heuristic
            end
            push!(stack, (new_heuristic, new_subset))    
        end
    end
    stack
end
