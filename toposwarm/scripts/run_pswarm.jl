"""
run_pswarm.jl  —  TopoSwarm projection step, called from the Quarto paper.

Usage:
    julia --project=toposwarm toposwarm/scripts/run_pswarm.jl <bridge.npz> <result.npz>

Reads:
    bridge.npz   node_weights (n_nodes × d), hit_map (n_nodes,),
                 bmu_indices (N,), labels (n_nodes,) [optional, per-node majority class]

Writes:
    result.npz   sample_rows (N,), sample_cols (N,), umatrix (rows × cols),
                 stress (scalar), node_rows (n_nodes,), node_cols (n_nodes,)
"""

include(joinpath(@__DIR__, "../src/TopoSwarm.jl"))
using .TopoSwarm
using NPZ

length(ARGS) == 2 || error("usage: run_pswarm.jl <bridge.npz> <result.npz>")
bridge_path, result_path = ARGS

println("Reading bridge: $bridge_path")
d = npzread(bridge_path)

node_weights = Float64.(d["node_weights"])   # (n_nodes, d)
hit_map      = Float64.(d["hit_map"])        # (n_nodes,)
bmu_indices  = Int.(d["bmu_indices"])        # (N,) 0-based

# per-node majority labels, pre-aggregated in Python export_npz (-1 = unoccupied)
labels = haskey(d, "labels") ? [v == -1 ? "" : string(v) for v in Int.(d["labels"])] : nothing

println("Building distance matrix  ($(size(node_weights, 1)) nodes × $(size(node_weights, 2)) features)")
D = pairwise_euclidean(node_weights)

pop_w  = hit_map ./ sum(hit_map)

println("Running TopoSwarm...\n")
result = pswarm!(D; pop_weights=pop_w, seed=42)

println("\n── Convergence ─────────────────────────────────────────────────")
show_stress(result.stress_history)

println("── Projection ──────────────────────────────────────────────────")
show_grid(result; labels=labels)

println("Computing U-matrix...")
U = umatrix(result)

println("Assigning raw samples to grid...")
node_rows = Int.(real.(result.positions))
node_cols = Int.(imag.(result.positions))
sample_rows, sample_cols = assign_raw_samples(result, bmu_indices)

npzwrite(result_path, Dict(
    "sample_rows" => sample_rows,
    "sample_cols" => sample_cols,
    "node_rows"   => node_rows,
    "node_cols"   => node_cols,
    "umatrix"     => U,
    "stress"      => [result.final_stress],
))

println("\nDone. Final stress: $(round(result.final_stress, digits=4))")
println("Result written to: $result_path")
