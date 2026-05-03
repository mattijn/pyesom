"""
run_pswarm_live.jl  —  watch the swarm self-organise in real time in your terminal.

Usage:
    julia --project=toposwarm toposwarm/scripts/run_pswarm_live.jl [dataset] [fps]

    dataset  : toy (default) | atom | chainlink | hepta | twodiamonds | wingnut | lsun
    fps      : frames per second, default 4

Examples:
    julia --project=toposwarm toposwarm/scripts/run_pswarm_live.jl
    julia --project=toposwarm toposwarm/scripts/run_pswarm_live.jl hepta 6
"""

include(joinpath(@__DIR__, "../src/TopoSwarm.jl"))
using .TopoSwarm
using Printf
using Statistics

dataset = length(ARGS) >= 1 ? ARGS[1] : "toy"
fps     = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4

if dataset == "toy"
    X = Float64[
        1.0  4.0;
        1.5  4.5;
        0.8  3.8;
        5.0  4.0;
        5.5  4.3;
        4.8  3.7;
        3.0  0.5;
        3.3  0.8;
        2.8  0.3]
    labels = ["A","A","A","B","B","B","C","C","C"]
    D = pairwise_euclidean(X)
else
    using NPZ
    fcps_path = joinpath(@__DIR__, "../../tests/fixtures/fcps.npz")
    isfile(fcps_path) || error("FCPS fixture not found: $fcps_path")
    z      = npzread(fcps_path)
    haskey(z, "$(dataset)_data") || error("Unknown dataset '$dataset'. Choose: atom chainlink hepta twodiamonds wingnut lsun")
    X      = Float64.(z["$(dataset)_data"])
    cls    = Int.(vec(z["$(dataset)_cls"]))
    labels = string.(cls)
    # z-score each feature so no axis dominates
    X = (X .- mean(X, dims=1)) ./ (std(X, dims=1) .+ 1e-9)
    D = pairwise_euclidean(X)
end

println("Dataset : $dataset   ($(size(D,1)) points)   fps=$fps")
println("Press Ctrl-C to abort early.\n")

result = pswarm_live!(D; labels=labels, seed=42, fps=fps)
