struct HVal{T}
    l::Int64
    h::T
end

Base.show(io::IO, s::HVal) = println(io, "HVal(l = ",s.l," h = ", s.h, ")")

function Base.isless(a::HVal, b::HVal)    a.h < b.h && return(true)
    a.h > b.h && return(false)
    return(a.l < b.l)
end


function accuracy_sdp(ŷ::AbstractMatrix, y::AbstractVector)
    correct = .!maximum(ŷ .!= y, dims = 1)
    mean(correct)
end

function accuracy_sdp(ŷ::CuMatrix, y::CuVector, omask::SBitSet)
    omask = cu(collect(omask))
    match = ŷ .!= y
    correct = .!maximum(match[omask,:], dims = 1)
    mean(correct)
end

function accuracy_sdp(ŷ::Matrix, y::Vector, omask::SBitSet)
    omask = collect(omask)
    match = ŷ .!= y
    correct = .!maximum(match[omask,:], dims = 1)
    mean(correct)
end

function accuracy_sdp(logits::AbstractMatrix, y::Integer)
    ŷ =  Flux.onecold(logits)
    sum(==(y), ŷ) / length(ŷ)
end


function accuracy_sdp(ii::SBitSet, sm, sampler, num_samples; verbose = false)
    r = condition(sampler, sm.input, ii)
    x = sample_all(r, num_samples)
    accuracy_sdp(sm.nn(x), sm.output)
end

function isvalid_sdp(ii::SBitSet, sm, ϵ, sampler, num_samples; verbose = false)
    acc = accuracy_sdp(ii, sm, sampler, num_samples)
    o = acc ≥ ϵ
    verbose && println("accuracy  = ",acc , " threshold = ", ϵ, " isvalid = ", o)
    o
end

function isvalid_sdp(ii::SBitSet, sm, sampler, num_samples; verbose = false)
    acc = accuracy_sdp(ii, sm, sampler, num_samples)
    verbose && println("accuracy  = ",acc)
    acc
end

function heuristic_sdp(ii::SBitSet, sm, ϵ, sampler, num_samples; verbose = false)
    acc = accuracy_sdp(ii, sm, sampler, num_samples)
    h = ϵ - acc
    verbose && println("heuristic = ", h)
    max(h, 0)
end

"""
    iscorrect(ŷ, y)

    return a row vector identifying which instance was correctly classified
"""
function iscorrect(ŷ::AbstractMatrix, y::Integer)
    transpose(Flux.onecold(ŷ) .== y)
end


function iscorrect(ŷ::CuMatrix, y::CuVector, omask::SBitSet)
    omask = cu(collect(omask))
    match = ŷ .!= y
    .!maximum(match[omask,:], dims = 1)
end

function iscorrect(ŷ::Matrix, y::Vector, omask::SBitSet)
    omask = collect(omask)
    match = ŷ .!= y
    .!maximum(match[omask,:], dims = 1)
end

function iscorrect(ŷ::AbstractMatrix, y::AbstractVector)
    .!maximum(ŷ .!= y, dims = 1)
end



"""
    (c, n) = batch_matches(xₛ, x, correct)

    input:    
    xₛ --- the input 
    x  --- perturbed output
    correct[i] --- indicates if outputput of the model on x[:,i] matches the correct output

    output:
    c --- how many times xₛ[i] == x[i] and the output is correct
    n --- how many times xₛ[i] == x[i]
"""
function batch_matches(xₛ::AbstractVector, x::AbstractMatrix, correct::AbstractMatrix)
    n = sum(x .== xₛ, dims = 2)

    _f(xᵢ, xₛᵢ, yᵢ) = xᵢ == xₛᵢ ? yᵢ : false
    c = sum(_f.(x, xₛ, correct), dims = 2)

    return(c, n)
end

function batch_sampl_heuristic(ii::SBitSet, sm, sampler, num_samples; verbose = false)
    r = condition(sampler, sm.input, ii)
    x = sample_all(r, num_samples)
    c, n = batch_matches(sm.input, x, iscorrect(argmax(sm.nn(x), dims=1), sm.output))
    # c, n = batch_matches(sm.input, x, iscorrect(sm.nn(x), sm.output))
    return vec(c), vec(n)
end


"""
    batch_heuristic(ii::SBitSet, sm, sampler, num_samples; verbose = false)

    Frequentist heuristic equal to ratio of correct classification to number of trials
"""
function batch_heuristic(ii::SBitSet, sm, sampler, num_samples; verbose = false)
    r = condition(sampler, sm.input, ii)
    x = sample_all(r, num_samples)
    batch_heuristic(sm.input, x, iscorrect(sm.nn(x), sm.output))
end

function batch_heuristic(xₛ, x, correct)
    c, n = batch_matches(xₛ, x, correct)
    map(cpu(c),cpu(n)) do cᵢ, nᵢ
        nᵢ == 0 ? 0.0 : cᵢ / nᵢ
    end
end

"""
    batch_beta(xₛ, x, correct, ϵ)
    batch_beta(ii::SBitSet, sm, sampler, num_samples, ϵ; verbose = false)

    Bayesian heuristic computing the probability that the accuracy is higher 
    than `ϵ.` The heuristic assumes Beta distribution of the posterior with 
    prior `Beta(1,1).`

"""
function batch_beta(xₛ, x, correct, ϵ)
    c, n = batch_matches(xₛ, x, correct)
    map(cpu(c),cpu(n)) do cᵢ, nᵢ
        logcdf(Beta(nᵢ - cᵢ + 1, cᵢ + 1), ϵ)
    end
end

function batch_beta(ii::SBitSet, sm, sampler, num_samples, ϵ; verbose = false)
    r = condition(sampler, sm.input, ii)
    x = sample_all(r, num_samples)
    batch_beta(sm.input, x, iscorrect(sm.nn(x), sm.output), ϵ)
end

struct BatchHeuristic{F,S,P}
    scalar::S
    batch::F
    finalize::P
end

Base.show(io::IO, s::BatchHeuristic) = println(io, "BatchHeuristic")

(sp::BatchHeuristic)(ii) = sp.finalize(sp.scalar(ii))

function ProbAbEx.expand_frwd(sm::ProbAbEx.Subset_minimal, stack, closed_list, ii::SBitSet, hfun::BatchHeuristic)
    acc = @timeit to "heuristic" hfun.batch(ii)
    for i in setdiff(1:sm.dims, ii)
        new_subset = push(ii, i)
        if new_subset ∉ closed_list
            push!(stack, (hfun.finalize(acc[i]), new_subset))
        end
    end
    stack
end


function ProbAbEx.expand_frwd(sm::ProbAbEx.Subset_minimal, stack, closed_list, ii::Tuple, heuristic_fun::BatchHeuristic)
    sp = heuristic_fun
    (I3, I2, I1) = ii

    accs = sp.scalar(ii)
    svals = sp.batch(ii)

    for i in setdiff(1:sm.dims[1], I3)
        new_subset = (push(I3, i), I2, I1)
        if new_subset ∉ closed_list
            # h = sp.finalize((;I₃ = svals.I₃[i], I₂ = accs.I₂, I₁ = accs.I₁))
            h = -(svals.I₃[i] + accs.I₂ + accs.I₁)
            # println("pushing1: ", h)
            push!(stack, (h, new_subset))
        end
    end

    for i in setdiff(1:sm.dims[2], I2)
        new_subset = (I3, push(I2, i), I1)
        if new_subset ∉ closed_list
            # h = sp.finalize((;I₃ = accs.I₃, I₂ = svals.I₂[i], I₁ = accs.I₁))
            h = -(accs.I₃ + svals.I₂[i] + accs.I₁)
            push!(stack, (h, new_subset))
        end
    end
    
    for i in setdiff(1:sm.dims[3], I1)
        new_subset = (I3, I2, push(I1, i))
        if new_subset ∉ closed_list
            # h = sp.finalize((;I₃ = accs.I₃, I₂ = accs.I₂, I₁ = svals.I₁[i]))
            h = -(accs.I₃ + accs.I₂ + svals.I₁[i])
            # println("pushing2: ", h)
            push!(stack, (h, new_subset))
        end
    end
    stack
end


#########################################################################################
# The code below deals with extensions to the search for minimal subset of neural network
#########################################################################################

function restrict_output(sm::Subset_minimal, ii)
    Subset_minimal(restrict_output(sm.nn, ii), sm.input, restrict_output(sm.output, ii))
end

function restrict_output(m::Chain, ii)
    Chain(m[1:end-1]..., restrict_output(m[end], ii))
end

function restrict_output(l::Dense, ii)
    Dense(restrict_output(l.weight, ii), restrict_output(l.bias, ii), l.σ)
end

restrict_output(w::Matrix, ii::SBitSet) = restrict_output(w, collect(ii))
restrict_output(w::Matrix, ii::Vector) = w[ii, :]
restrict_output(w::CuMatrix, ii::Vector) = w[cu(ii), :]
restrict_output(w::CuMatrix, ii::SBitSet) = restrict_output(w, collect(ii))
restrict_output(w::Vector, ii::SBitSet) = restrict_output(w, collect(ii))
restrict_output(w::Vector, ii::Vector) = w[ii]
restrict_output(w::CuVector, ii::Vector) = w[cu(ii)]
restrict_output(w::CuVector, ii::SBitSet) = restrict_output(w, collect(ii))
restrict_output(x::Integer, ii) = x


function accuracy_sdp3(ii::Tuple, sm::Subset_minimal, samplers, num_samples; verbose = false)
    (I3, I2, I1) = ii
    xₛ = sm.input
    h₃_h₂ = isvalid_sdp(I3, restrict_output(Subset_minimal(sm.nn[1], xₛ), I2), samplers[1], num_samples)
    h₂_h₁ = isvalid_sdp(I2, restrict_output(Subset_minimal(sm.nn[2], sm.nn[1](xₛ)), I1), samplers[2], num_samples)
    # println("AAAAAAAAAAAAAA")
    h₁_h₀ = isvalid_sdp(I1, Subset_minimal(sm.nn[3], sm.nn[1:2](xₛ), argmax(sm.output)), samplers[3], num_samples)
    # println("sm.output: ", argmax(sm.output))
    h₃_h₂ = isempty(I2) ? 0.0 : h₃_h₂
    h₂_h₁ = isempty(I1) ? 0.0 : h₂_h₁
    return(I₃ = h₃_h₂, I₂ = h₂_h₁, I₁ = h₁_h₀)
end

function isvalid_sdp3(ii::Tuple, sm, ϵ, samplers, num_samples; verbose = false)
    accs = accuracy_sdp3(ii, sm, samplers, num_samples; verbose)
    verbose && println("accuracy  = ", accs)
    all(e ≥ ϵ for e in accs)
end

# function batch_heuristic3(ii::Tuple, sm::Subset_minimal, samplers, num_samples; verbose = false)
#     (I3, I2, I1) = ii
#     xₛ = sm.input
#     h₃_h₂ = shapley_heuristic(I3, restrict_output(Subset_minimal(sm.nn[1], xₛ), I2), samplers[1], num_samples)
#     h₂_h₁ = shapley_heuristic(I2, restrict_output(Subset_minimal(sm.nn[2], sm.nn[1](xₛ)), I1), samplers[2], num_samples)
#     h₁_h₀ = shapley_heuristic(I1, Subset_minimal(sm.nn[3], sm.nn[1:2](xₛ)), samplers[3], num_samples)
#     if isempty(I2)
#         h₃_h₂ .= 0.0
#     end    
#     if isempty(I1)
#         h₂_h₁ .= 0.0
#     end    
#     return(I₃ = h₃_h₂, I₂ = h₂_h₁, I₁ = h₁_h₀)
# end

function shapley_heuristic(ii::Tuple, sm::Subset_minimal, sampler, num_samples; verbose = false)
    (I3, I2, I1) = ii

    h₃ = sm.input
    h₂ = sm.nn[1](h₃)
    h₁ = sm.nn[2](h₂)

    jj = empty_sbitset(sum(sm.dims))
    jj = foldl((r,i) -> push(r, i), I3, init = jj)
    jj = foldl((r,i) -> push(r, i + sm.dims[1]), I2, init = jj)
    jj = foldl((r,i) -> push(r, i + sm.dims[1] + sm.dims[2]), I1, init = jj)


    r = condition(sampler, vcat(h₃, h₂, h₁), jj)
    x = sample_all(r, 10_000)

    x₃ = x[1:sm.dims[1],:]
    x₂ = x[sm.dims[1]+1:sm.dims[1]+sm.dims[2],:]
    x₁ = x[sm.dims[1]+sm.dims[2]+1:end,:]

    # this checks that each inner layer can explain the output
    (;  h₃ = shapley(h₃, x₃, iscorrect(sm.nn(x₃), sm.output)),
        h₂ = shapley(h₂, x₂, iscorrect(sm.nn[2:3](x₂), sm.output)),
        h₁ = shapley(h₁, x₁, iscorrect(sm.nn[3](x₁), sm.output)),

        # the input to the neural network has to imply h₃[I2] and h₂[I1] 
        h₃_h₂ = shapley(h₃, x₃, iscorrect(sm.nn[1](x₃), h₂, I2)),
        h₃_h₁ = shapley(h₃, x₃, iscorrect(sm.nn[1:2](x₃), h₁, I1)),
        h₂_h₁ = shapley(h₂, x₂, iscorrect(sm.nn[2](x₂), h₁, I1)),
    )
end

function batch_heuristic3(ii::Tuple, sm::Subset_minimal, samplers, num_samples; verbose = false)
    (I3, I2, I1) = ii
    xₛ = sm.input
    h₃_h₂ = batch_heuristic(I3, restrict_output(Subset_minimal(sm.nn[1], xₛ), I2), samplers[1], num_samples)
    h₂_h₁ = batch_heuristic(I2, restrict_output(Subset_minimal(sm.nn[2], sm.nn[1](xₛ)), I1), samplers[2], num_samples)
    h₁_h₀ = batch_heuristic(I1, Subset_minimal(sm.nn[3], sm.nn[1:2](xₛ)), samplers[3], num_samples)
    if isempty(I2)
        h₃_h₂ .= 0.0
    end    
    if isempty(I1)
        h₂_h₁ .= 0.0
    end  
  
    return((I₃ = h₃_h₂, I₂ = h₂_h₁, I₁ = h₁_h₀))
end

"""
    shapley_heuristic2(ii::Tuple, sm::Subset_minimal, sampler, num_samples; verbose = false)

    heuristic designed to estimate accuracy_sdp2, but it uses joint_sampler instead of 
    sampling only from the input distribution. We hope that by doing so, we sample
    more interesting distributions.
"""
function shapley_heuristic2(ii::Tuple, sm::Subset_minimal, sampler, num_samples; verbose = false)
    (I3, I2, I1) = ii

    xₛ₃ = sm.input
    xₛ₂ = sm.nn[1](xₛ₃)
    xₛ₁ = sm.nn[2](xₛ₂)
    xₛ₀ = sm.nn[3](xₛ₁)

    jj = empty_sbitset(sum(sm.dims))
    jj = foldl((r,i) -> push(r, i), I3, init = jj)
    jj = foldl((r,i) -> push(r, i + sm.dims[1]), I2, init = jj)
    jj = foldl((r,i) -> push(r, i + sm.dims[1] + sm.dims[2]), I1, init = jj)


    r = condition(sampler, vcat(xₛ₃, xₛ₂, xₛ₁), jj)
    x = sample_all(r, 10_000)

    x₃ = x[1:sm.dims[1],:]
    x₂ = x[sm.dims[1]+1:sm.dims[1]+sm.dims[2],:]
    x₁ = x[sm.dims[1]+sm.dims[2]+1:end,:]


    # Then the classification should imply the first hidden units
    h₂ = sm.nn[1](x₃)
    mask₃ = iscorrect(h₂, xₛ₂, I2)
    h₃_h₂ = shapley(xₛ₃, x₃, mask₃)

    # Then these correct should imply the second row
    h₁ = sm.nn[2](h₂)
    mask₂ = iscorrect(h₁, xₛ₁, I1)
    mc = vec(mask₃)
    h₂_h₁ = shapley(xₛ₂, h₂[:,mc], mask₂[:,mc])
    # h₂_h₁ = shapley(xₛ₂, h₂, mask₂)

    h₀ = sm.nn[3](h₁)
    mask₁ = iscorrect(h₀, xₛ₀)
    mc = vec(mask₃ .* mask₂)
    # h₁_h₀ = shapley(xₛ₁, h₁, mask₁)
    h₁_h₀ = shapley(xₛ₁, h₁[:,mc], mask₁[:,mc])

    (;h₃_h₂, h₂_h₁, h₁_h₀)
end

function ProbAbEx.expand_bcwd(sm::ProbAbEx.Subset_minimal, stack, closed_list, ii::Tuple, heuristic_fun::BatchHeuristic)
    sp = heuristic_fun
    (I3, I2, I1) = ii
    for i in I3
        new_subset = (pop(I3, i), I2, I1)
        if new_subset ∉ closed_list
            hvals = accuracy_sdp2(new_subset, sp.sm, sp.sampler, sp.num_samples; verbose =  sp.verbose)
            hval = agg(hvals...)
            push!(stack, (hval, hval, new_subset))
        end
    end

    for i in I2
        new_subset = (I3, pop(I2, i), I1)
        if new_subset ∉ closed_list
            hvals = accuracy_sdp2(new_subset, sp.sm, sp.sampler, sp.num_samples; verbose =  sp.verbose)
            hval = agg(hvals...)
            push!(stack, (hval, hval, new_subset))
        end
    end
    
    for i in I1
        new_subset = (I3, I2, pop(I1, i))
        if new_subset ∉ closed_list
            hvals = accuracy_sdp2(new_subset, sp.sm, sp.sampler, sp.num_samples; verbose =  sp.verbose)
            hval = agg(hvals...)
            push!(stack, (hval, hval, new_subset))
        end
    end
    stack
end

depth_first(ii::SBitSet) = length(ii)
depth_first(ii::Tuple)  = mapreduce(length, +, ii)





function shapley_heuristic(ii::SBitSet, sm, sampler, num_samples, verbose = false)
    r = condition(sampler, sm.input, ii)
    x = sample_all(r, num_samples)
    # y = Flux.onecold(sm.nn(x)) .== sm.output
    y = argmax(sm.output)
    scores = sm.nn(x)
    if size(scores, 1) == 0 || size(scores, 2) == 0
        error("Empty prediction scores — cannot compute onecold.")
    end
    ŷ = Flux.onecold(scores)
    sum(==(y), ŷ) / length(ŷ)
end

function shapley_heuristic(ii::Tuple, sm, sampler, num_samples; verbose = false)
    (I3, I2, I1) = ii
    xₛ = sm.input
    # println("shapley_heuristic")
    h₃ = shapley_heuristic(I3, Subset_minimal(sm.nn, xₛ), sampler, num_samples)
    h₂ = shapley_heuristic(I2, Subset_minimal(sm.nn[2:3], sm.nn[1](xₛ)), sampler, num_samples)
    h₁ = shapley_heuristic(I1, Subset_minimal(sm.nn[3], sm.nn[1:2](xₛ)), sampler, num_samples)

    h₃_h₂ = 0.0
    h₂_h₁ = 0.0

    if !isempty(I2)
        h₃_h₂ = shapley_heuristic(I3, restrict_output(Subset_minimal(sm.nn[1], xₛ), I2), samplers[1], num_samples)
    end 
    if !isempty(I1)
        h₂_h₁ = shapley_heuristic(I2, restrict_output(Subset_minimal(sm.nn[2], sm.nn[1](xₛ)), I1), samplers[2], num_samples)
    end    
    h₁_h₀ = shapley_heuristic(I1, Subset_minimal(sm.nn[3], sm.nn[1:2](xₛ)), samplers[3], num_samples)

        # the input to the neural network has to imply h₃[I2] and h₂[I1] 
        # h₃_h₂ = shapley_heuristic(I3, restrict_output(Subset_minimal(sm.nn[1], xₛ), I2), sampler, num_samples),
        # h₃_h₁ = shapley_heuristic(I3, restrict_output(Subset_minimal(sm.nn[2], sm.nn[1](xₛ)), I1), sampler, num_samples),
        # h₂_h₁ = shapley_heuristic(I2, (Subset_minimal(sm.nn[3], sm.nn[1:2](xₛ)), I1), sampler, num_samples),
    #
    hs = (
        h₃ = h₃,
        h₂ = h₂,
        h₁ = h₁,
        h₃_h₂ = h₃_h₂,
        h₂_h₁ = h₂_h₁,
        h₁_h₀ = h₁_h₀,
    )
    (;
    hsum = mapreduce(x -> max(0, 0.99 - x), +, hs),
    hmax = mapreduce(x -> max(0, 0.99 - x), max, hs),
    )
end

struct ShapleyHeuristic{S,SM}
    sm::SM
    sampler::S
    num_samples::Int
    verbose::Bool
end

function ShapleyHeuristic(sm, sampler, num_samples, verbose = false)
    ShapleyHeuristic(sampler, sm, num_samples, verbose)
end

(sp::ShapleyHeuristic)(ii::SBitSet) = heuristic_sdp(ii, sp.sm, 0.99, sp.sampler, sp.num_samples)
(sp::ShapleyHeuristic)(ii::Tuple) = shapley_heuristic(ii, sp.sm, sp.sampler, sp.num_samples)


