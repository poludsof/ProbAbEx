# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ProbAbEx is a Julia research package for **probabilistic abductive explanations** of neural-network classifiers (MNIST, with in-progress CelebA work). Given an input `xₛ` and a classifier, it searches for a minimal subset of input features (pixels or superpixels) such that, when those features are fixed and the rest are resampled from a conditional distribution, the classifier keeps its prediction with probability ≥ `1 - ϵ` (the "SDP" criterion).

## Running code

There is no test suite. `tests/` and `scripts/` hold interactive driver scripts meant to be run in a REPL with Revise, cell by cell (`##` separates cells). Nothing there is an automated test.

```bash
julia --project=.            # from repo root; then e.g. include("tests/vaeac_search.jl")
julia --project=. tests/run_vaeac_training.jl   # train the VAEAC sampler, saves to models/
```

The GPU path runs on a SLURM cluster (`srun -p gpufast --gres=gpu:1 --mem=16000 --pty bash -i`). Scripts call `Reactant.set_default_backend("gpu")` and `reactant_device(; force=true)`.

Scripts load the module as `import ProbAbEx as PAE` and run from the repo root. Paths are relative, like `models/...`, or built with `joinpath(@__DIR__, "..", "models", ...)`.

## Stack and migration state

The project is moving from **Flux + CUDA.jl** to **Lux + Reactant (XLA, `@compile`) + Enzyme**.
- Current code paths use Lux/Reactant: `tests/vaeac_search.jl`, `tests/run_vaeac_training.jl`, `src/samplers/VAEAC_*.jl`, and the `*_batched` functions in `src/heuristics_criteria.jl`.
- Older scripts still use Flux/CUDA (`tests/debug.jl`, `tests/user_documentation.jl`, `tests/script.jl`, `ext/CUDAExt.jl`). Some even reference the old package name `Subset_minimal_search`. Treat these as stale.
- `ext/` is **not** registered as package extensions in `Project.toml`. Scripts pull `ext/ReactantExt.jl` in by hand with `include(...)`, using a hardcoded absolute path.
- Several `include`s in `src/ProbAbEx.jl` are commented out (`milp.jl`, `plots.jl`, `datasets/celeba.jl`). Files such as `find_explanations.jl`, `kl_boundaries.jl`, `visualization*.jl`, `forward_beamsearch.jl` and `samplers/celeba_VAEAC.jl` are not included in the module at all.

## Architecture

**Problem object**: `Subset_minimal(nn, input, output, dims)` is defined in `src/ProbAbEx.jl`. `nn` is any callable, often a closure over a Reactant-compiled Lux model. Feature subsets are `StaticBitSets.SBitSet{N,UInt64}` with `N = ceil(n/64)`. Older code also uses `(I3, I2, I1)` tuples of bitsets for explanations across several layers.

**Search algorithms** (`forward_search.jl`, `backward_search.jl`, `beam_search.jl`) share one signature:
`search(sm, ii, isvalid::Function, heuristic_fun; time_limit, terminate_on_first_solution, ...)`.
- `isvalid(ii) -> Bool` checks the SDP threshold.
- `heuristic_fun(ii)` returns a score. Forward search sorts its stack by `-h` and then calls `pop!`, so it expands the subset with the **lowest** `h` first. `heuristic_sdp*` returns `max(ϵ - acc, 0)`, which means "how far below the threshold".
- Forward search can refine each solution with backward search (`refine_with_backward`).
- All three use `sm.dims` to enumerate candidate features.

**Samplers** (`src/samplers/`) share an informal interface: `condition(sampler, xₛ, known_ii::SBitSet) -> Conditioned*` followed by `sample_all(conditioned, n)`. Implementations:
- `UniformDistribution`, which samples ±1 pixels.
- `BernoulliMixture`, which loads its centers from `models/milan_centers.jls`.
- `VAEACSampler`, a VAE with arbitrary conditioning trained in `VAEAC_training.jl` (prior, encoder and decoder networks; loaded with `load_vaeac_sampler(path, dev)`). This is the current focus.

**Criteria and heuristics** live in `src/heuristics_criteria.jl` and older `criterium.jl`. Pixel encoding differs by path: the VAEAC path uses `{0,1}` pixels, while the Uniform and Bernoulli paths use `{-1,+1}` (`preprocess_bin_neg`). The current GPU pipeline works as follows:
1. `build_compiled_accuracy_looped(...)` builds a `CompiledAccLooped`. This single Reactant-compiled kernel samples from VAEAC, classifies the samples and averages the accuracy over `n_batches × batch_size` samples. It stays on the GPU to avoid GPU-to-CPU transfers.
2. `isvalid_sdp_batched` and `heuristic_sdp_batched` wrap that kernel. `ShapleyHeuristic(sm, sampler, ϵ, compiled_acc, model_cls, ps, st)` is the callable heuristic.
3. Superpixel variant: `grid_superpixels` and `build_sp_map` define the superpixels. `SPSubsetMinimal` sets `dims = n_sp`. `make_sp_isvalid` and `SPShapleyHeuristic` expand a superpixel bitset to pixels before evaluating it.

Many functions have several overloads that differ only in argument lists, because of older and newer pipelines. Check which overload a script actually calls before editing.

**Timing**: the module-level `const to = TimerOutput()` (`PAE.to`) is used through `@timeit` throughout the search code. Driver scripts call `reset_timer!(PAE.to)` and `show(PAE.to)`.

## Artifacts

- `models/*.jls`: serialized tuples `(model, ps, st)`. Parameters are moved to the CPU before `serialize`. Examples are `mnist_conv_model.jls` (the classifier) and `small_new_vaeac.jls` (the VAEAC).
- `training_plots/`: training logs and plots.
- `result_*`: output images.
- CelebA images are expected under `src/datasets/img_align_celeba/` (gitignored).
