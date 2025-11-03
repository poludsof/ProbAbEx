"""

    The design is that stack will contain only valid solutions.
    In every step, I will pop the best solution and filter all its childs 
    to find those, which are valid. If there is no valid child, than the
    sample is subset_minimal and I add it to the solutions.
    If there is at least one valid child, I add it to the stack.

    I think that the heuristic function should be length, as I want to be expanding 
    solutions with minimal length.
"""
function backward_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun; time_limit=Inf, terminate_on_first_solution=true) where {TT}
    # if !isvalid(ii) 
    #     println("initial verification of the solution failed, returning the solution")
    #     return ii
    # end
    println("sm:::::::", sm.output)
    println("II length: ", solution_length(ii))
    initial_hsum, initial_hmax = heuristic_fun(ii)
    println(" Initial heuristic: ", initial_hsum, ", ", initial_hmax)

    stack = [(initial_hsum, initial_hmax, ii)]
    closed_list = Set{TT}()

    solutions = Set{TT}()
    steps = 0
    start_time = time()
    new_subsets = TT[]

    println("Starting backward search...")
    @timeit to "backward search" while !isempty(stack)
        steps += 1

        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solution")
            return solutions
        end

        sort!(stack, by = ii -> length(ii[end]), rev = true) # we sort of want to expand minimum length first
        current_hsum, current_hmax, ii = pop!(stack)
        closed_list = push!(closed_list, ii)

        new_subsets = expand_bcwd(sm, stack, closed_list, ii, heuristic_fun)
        new_subsets = filter(s -> isvalid(s[3]), new_subsets)
        if isempty(new_subsets)
            push!(solutions, ii)
            terminate_on_first_solution && return solutions
        else
            println("step: $steps, length $(solution_length(ii)) with hsum: $current_hsum, hmax: $current_hmax")
            append!(stack, new_subsets)
        end
    end
    println("Stack is empty")
    last_solution = collect(solutions)[end]
    if isempty(solutions)
        println("No solution found")
        push!(solutions, ii)
    end
    return solutions # the best
end

new_subsets_bcwd(ii::SBitSet, idim) = [pop(ii, i) for i in 1:idim if i in ii]

function new_subsets_bcwd((I3, I2, I1)::T, idims::Tuple) where {T<:Tuple}
    new_subsets = T[]
    if I3 !== nothing
        append!(new_subsets, [(pop(I3, i), I2, I1) for i in 1:idims[1] if i in I3])
    end
    if I2 !== nothing
        append!(new_subsets, [(I3, pop(I2, i), I1) for i in 1:idims[2] if i in I2])
    end
    if I1 !== nothing
        append!(new_subsets, [(I3, I2, pop(I1, i)) for i in 1:idims[3] if i in I1])
    end
    new_subsets
end

function expand_bcwd(sm::Subset_minimal, stack, closed_list, ii, heuristic_fun)
    for new_subset in new_subsets_bcwd(ii, sm.dims)
        if new_subset ∉ closed_list
            new_hsum = 0
            new_hmax = 0
            # new_hsum, new_hmax = @timeit to "heuristic" heuristic_fun(new_subset)
            push!(stack, (new_hsum, new_hmax, new_subset))
        end
    end
    stack
end