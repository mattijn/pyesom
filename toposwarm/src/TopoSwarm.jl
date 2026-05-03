module TopoSwarm

using Random
using Statistics: mean, std
using Printf

export Grid, make_grid, valid_position,
       toroidal_dist, toroidal_dist_matrix,
       PolarGrid, setup_polar_grid, allowed_positions,
       Swarm, init_swarm, move_bot!, is_free,
       stress, stress_weighted, normalise_matrix, stress_if_moved,
       swarm_epoch!, swarm_radius!,
       PswarmResult, pswarm!, pswarm_live!,
       projection_data, umatrix, umatrix_interpolated,
       assign_raw_samples,
       show_grid, show_stress,
       zscore, pairwise_euclidean,
       show_data, show_dist

# ── Data utilities ───────────────────────────────────────────────

"""
    zscore(X) -> Matrix{Float64}

Column-wise z-score: subtract column mean, divide by column std.
Makes all features live on the same scale before computing distances.
"""
function zscore(X::Matrix{Float64}) :: Matrix{Float64}
    mu = mean(X, dims=1)
    sg = std(X,  dims=1)
    return (X .- mu) ./ (sg .+ 1e-9)
end

"""
    pairwise_euclidean(X) -> Matrix{Float64}

Symmetric n×n Euclidean distance matrix from an n×d feature matrix.
The diagonal is zero; entry [i,j] is the Euclidean distance between rows i and j.
"""
function pairwise_euclidean(X::Matrix{Float64}) :: Matrix{Float64}
    n = size(X, 1)
    D = zeros(n, n)
    for i in 1:n
        for j in (i+1):n
            d = sqrt(sum((X[i, :] .- X[j, :]) .^ 2))
            D[i, j] = d
            D[j, i] = d
        end
    end
    return D
end

# ── ASCII data visualisation ──────────────────────────────────────

"""
    show_data(X, labels; title, width, height) -> Nothing

ASCII 2D scatter plot of an n×2 matrix X.
The first character of each label is placed at the point's grid cell.
Higher y values appear higher in the plot (natural orientation).
"""
function show_data(X::Matrix{Float64}, labels::Vector{String};
                   title::String = "Data",
                   width::Int    = 36,
                   height::Int   = 14) :: Nothing
    size(X, 2) >= 2 || error("show_data needs at least 2 columns in X")
    n  = size(X, 1)
    xs = X[:, 1]
    ys = X[:, 2]

    xlo, xhi = minimum(xs), maximum(xs)
    ylo, yhi = minimum(ys), maximum(ys)
    xhi ≈ xlo && (xhi = xlo + 1.0)
    yhi ≈ ylo && (yhi = ylo + 1.0)

    # map coordinates to grid cells; flip y so larger y is higher in the print
    ci = clamp.(round.(Int, (xs .- xlo) ./ (xhi - xlo) .* (width  - 1)) .+ 1, 1, width)
    ri = clamp.(round.(Int, (yhi .- ys) ./ (yhi - ylo) .* (height - 1)) .+ 1, 1, height)

    grid = fill(' ', height, width)
    for i in 1:n
        grid[ri[i], ci[i]] = labels[i][1]
    end

    println("$title  ($n points)\n")
    println("  ┌" * "─"^width * "┐")
    for r in 1:height
        print("  │")
        print(String(grid[r, :]))
        println("│")
    end
    println("  └" * "─"^width * "┘")
    println()
    return nothing
end

"""
    show_dist(D, labels; title) -> Nothing

ASCII heatmap of a pairwise distance matrix.
Each cell is rendered as a shade character encoding D[i,j]:

  ' '  ·   ░   ▒   ▓   █
  near ←——————————————→ far

When rows and columns are sorted by cluster, the block structure is visible:
light blocks on the diagonal (within-cluster), dark cells off-diagonal (between clusters).
"""
function show_dist(D::Matrix{Float64}, labels::Vector{String};
                   title::String = "Distance matrix") :: Nothing
    n  = size(D, 1)
    lo = minimum(D)
    hi = maximum(D)
    hi ≈ lo && (hi = lo + 1.0)

    short = [l[1:min(1, length(l))] for l in labels]

    println("$title  ($n × $n)\n")
    print("    ")
    for j in 1:n; print(short[j], " "); end
    println()
    for i in 1:n
        print(short[i], "   ")
        for j in 1:n
            print(_shade_char(D[i, j], lo, hi), " ")
        end
        println()
    end
    println()
    return nothing
end

# ── Step 1: Grid ─────────────────────────────────────────────────

"""
    Grid

Rectangular environment for the databots
Edges wrap left <> right and top <> bottom (Pac-Man style)
"""
struct Grid
    lines :: Int
    cols  :: Int
    function Grid(lines::Int, cols::Int)
        lines >= 3 || error("Grid needs at least 3 lines, got $lines")
        cols  >= 3 || error("Grid needs at least 3 cols,  got $cols")
        new(lines, cols)
    end
end

"""
    make_grid(n) -> Grid

Make me an envrionment big enough for n bots to move around and
self-organise without being cramped.
It creates an grid area which is ~3-5x larger than the n-amount of bots.
"""
function make_grid(n::Int) :: Grid
    side = ceil(Int, sqrt(n * 0.2) * 4)
    side = isodd(side) ? side : side + 1
    return Grid(side, side)
end

"""
    valid_position(g, pos) -> Bool

Make sure a position can be placed within the grid.
Position is a complete number so it can have both the row-position and col-position
in a single value (real-part: row/x, imaginary-part: col/y)
"""
function valid_position(g::Grid, pos::ComplexF64) :: Bool
    r, c = Int(real(pos)), Int(imag(pos))
    return 1 <= r <= g.lines && 1 <= c <= g.cols
end

# ── Step 2: Toroidal distance ─────────────────────────────────────

"""
    toroidal_dist(g, a, b) -> Float64

Pac-Man distance between two points in the grid.
Positions are complete numbers so it contains both the row-position and col-position
in a single value (real-part: row/x, imaginary-part: col/y)
"""
function toroidal_dist(g::Grid, a::ComplexF64, b::ComplexF64) :: Float64
    dx = abs(real(a) - real(b))
    dy = abs(imag(a) - imag(b))
    dx = min(dx, g.lines - dx + 1)
    dy = min(dy, g.cols  - dy + 1)
    return sqrt(dx^2 + dy^2)
end

"""
    toroidal_dist_matrix(g, positions) -> Matrix

Pac-Man distance between all bots as matrix.
"""
function toroidal_dist_matrix(g::Grid, positions::Vector{ComplexF64}) :: Matrix{Float64}
    n = length(positions)
    D = Matrix{Float64}(undef, n, n)
    for i in 1:n
        D[i, i] = 0.0
        for j in (i+1):n
            d = toroidal_dist(g, positions[i], positions[j])
            D[i, j] = d
            D[j, i] = d
        end
    end
    return D
end

# ── Step 3: Polar grid ────────────────────────────────────────────

struct PolarGrid
    rmin  :: Int
    rmax  :: Int

    # for each radius R: the set of (dr, dc) offsets that land on that ring
    # rings[R] = Vector of complete offsets (real=delta-row, imag=delta-col)
    rings :: Vector{Vector{ComplexF64}}
end

"""
    setup_polar_grid(g) -> PolarGrid

Result are ring coordinates for different r.
Number of r's and range of r's are based on grid size.
Each x below is a valid dx,dy for that particular r (first 3 radi are still squares).

r = 1 (8 positions)
x x x
x o x
x x x

r = 2 (16 positions)
x x x x x
x       x
x   o   x
x       x
x x x x x
"""
function setup_polar_grid(g::Grid) :: PolarGrid
    # maximum radius can maximum be half of the grid
    # minimum radius adds 1, if grid width adds 20
    # ÷ gives the round down Integer, different compare to /
    rmax = g.lines ÷ 2
    rmin = max(1, rmax ÷ 10)

    rings = Vector{Vector{ComplexF64}}(undef, rmax)

    # TODO: double check if this shall not be rmin:rmax
    for r in rmin:rmax
        offsets = ComplexF64[]
        for dr in -r:r, dc in -r:r
            dist = sqrt(dr^2 + dc^2)
            if floor(Int, dist) == r
                push!(offsets, ComplexF64(dr, dc))
            end
        end
        rings[r] = offsets
    end
    return PolarGrid(rmin, rmax, rings)
end

"""
    allowed_positions(pg, g, bot_pos, R) -> Vector

Given a position of a bot, returns the possible coordinates
it can go to in the main grid (in Pac-Man style) given a search radius.
It is able to do so using the delta coords for each radi of computed polar grids
"""
function allowed_positions(pg::PolarGrid, g::Grid,
        bot_pos::ComplexF64, R::Int) :: Vector{ComplexF64}

    row = real(bot_pos)
    col = imag(bot_pos)
    result = ComplexF64[]
    for offset in pg.rings[R]
        # eg for a grid of 3 by 3 and a bot position on 1,1
        # for offset -1 (eg. please bot, do one step left) will return in a
        # new_row of 3 (calculate: mod(1 - 1 + -1, 3) + 1)
        # so, like Pac-Man, the bot enters from the other side of the grid
        new_row = mod(row - 1 + real(offset), g.lines) + 1
        new_col = mod(col - 1 + imag(offset), g.cols)  + 1
        push!(result, ComplexF64(new_row, new_col))
    end
    return result
end

# ── Step 4: Swarm initialisation ─────────────────────────────────

"""
    Swarm

The population of DataBots on the grid.

pos contains one position per bot (real=row, imag=col)
occupied is a set of the positions, used to lookup of taken cells
n is the number of bots (== number of data points)

length(pos) == n  cannot be equal to length(occupied) == n
i.e. no two bots share a cell at any time.
"""
struct Swarm
    pos      :: Vector{ComplexF64}
    occupied :: Set{ComplexF64}
    n        :: Int
end

"""
    init_swarm(g, n, seed) -> Swarm

we shuffle all available coordinates in the grid.
take the first n coordinates; these can be used to place the bots.
"""
function init_swarm(g::Grid, n::Int; seed::Union{Int,Nothing}=nothing) :: Swarm
    n <= g.lines * g.cols || error("Too many bots ($n) for grid $(g.lines)×$(g.cols)")
    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)
    all_cells = [ComplexF64(r, c) for r in 1:g.lines for c in 1:g.cols]
    shuffle!(rng, all_cells)
    chosen = all_cells[1:n]
    return Swarm(chosen, Set(chosen), n)
end

"""
    move_bot!(swarm, idx, new_pos)

Based on the existing positions in swarm,
we add the new position and remove the previous position.
If the new position is taken already, then the positions merge
(in theory, in practice only free positions are considered).
We combine it into a single function so they don't deviate
"""
function move_bot!(swarm::Swarm, idx::Int, new_pos::ComplexF64)
    old_pos = swarm.pos[idx]
    delete!(swarm.occupied, old_pos)
    swarm.pos[idx] = new_pos
    push!(swarm.occupied, new_pos)
end

"""
    is_free(swarm, pos) -> Bool

Check if position is not in set of occupied cells.
If position is not in existing swarm positions, than return true
"""
function is_free(swarm::Swarm, pos::ComplexF64) :: Bool
    return pos ∉ swarm.occupied
end

# ── Step 5: Stress ────────────────────────────────────────────────

"""
    normalise_matrix(D) -> Matrix{Float64}

scale a distance matrix to [0,1] by dividing by its maximum.
applied to both D_data and D_grid before stress is computed
so the two matrices live on the same scale.
"""
function normalise_matrix(D::Matrix{Float64}) :: Matrix{Float64}
    m = maximum(D)
    m == 0.0 && return D
    return D ./ m
end

"""
    stress(D_data, D_grid) -> Float64

How unhappy the current bot placement is.

Both matrices are normalised to [0,1] first so they live on the same scale.
For every pair (i, j) we square the difference between the data distance and the
grid distance.  A perfect placement scores 0.0; higher means the grid layout
disagrees with what the data says about who is close to whom.
"""
function stress(D_data::Matrix{Float64}, D_grid::Matrix{Float64}) :: Float64
    size(D_data) == size(D_grid) || error("Distance matrices must have the same size")
    n      = size(D_data, 1)
    nd     = normalise_matrix(D_data)
    ng     = normalise_matrix(D_grid)
    total  = 0.0
    npairs = 0
    for i in 1:n
        for j in (i+1):n
            diff    = nd[i,j] - ng[i,j]
            total  += diff^2  # add a power to amplify stress for larger distances
            npairs += 1
        end
    end
    return total / npairs
end

"""
    stress_if_moved(D_data, g, swarm, bot_idx, candidate) -> Float64

Compute stress for the hypothetical state where bot_idx
has moved to candidate, without actually moving it.
Used in the jump decision loop.
"""
function stress_if_moved(D_data::Matrix{Float64},
                          g::Grid,
                          swarm::Swarm,
                          bot_idx::Int,
                          candidate::ComplexF64;
                          s_fn=(dd,dg)->stress(dd,dg)) :: Float64
    tmp          = copy(swarm.pos)
    tmp[bot_idx] = candidate
    D_grid       = toroidal_dist_matrix(g, tmp)
    return s_fn(D_data, D_grid)
end

# ── Step 6: Jump decision loop ────────────────────────────────────

"""
    swarm_epoch!(swarm, D_data, g, pg, R; rng) -> (n_jumped, stress)

One pass over a random subset of bots at search radius R.

The fraction of bots allowed to jump shrinks as R decreases — roughly 50 % at
Rmax (wide exploration) down to 5 % at Rmin (fine-tuning).  Each active bot
samples up to 4 candidate positions on the ring at radius R, moves to whichever
reduces stress, and stays put if none does.

Returns the number of bots that actually moved and the stress after all moves.
"""
function swarm_epoch!(swarm::Swarm,
                      D_data::Matrix{Float64},
                      g::Grid,
                      pg::PolarGrid,
                      R::Int;
                      rng=Random.default_rng(),
                      s_fn=(dd,dg)->stress(dd,dg)) :: Tuple{Int, Float64}

    # fraction of bots that get to jump shrinks with R
    # high R will have high number candidates (~50%), low R will have few candiates (~5%)
    # large R -> explore widely, small R -> fine-tune
    jump_fraction = 0.05 + 0.45 * (R - pg.rmin) / max(1, pg.rmax - pg.rmin)
    n_candidates  = max(1, round(Int, swarm.n * jump_fraction))
    bot_order     = shuffle(rng, 1:swarm.n)
    active        = bot_order[1:n_candidates]
    n_jumped      = 0

    for idx in active
        ring            = allowed_positions(pg, g, swarm.pos[idx], R)
        isempty(ring)  && continue
        candidates      = ring[rand(rng, 1:length(ring), min(4, length(ring)))]
        free_candidates = filter(c -> is_free(swarm, c), candidates)
        isempty(free_candidates) && continue

        current_s = stress_if_moved(D_data, g, swarm, idx, swarm.pos[idx]; s_fn)
        best_pos  = swarm.pos[idx]
        best_s    = current_s

        for cand in free_candidates
            s = stress_if_moved(D_data, g, swarm, idx, cand; s_fn)
            if s < best_s
                best_s   = s
                best_pos = cand
            end
        end

        # jump if we found something better
        if best_pos != swarm.pos[idx]
            move_bot!(swarm, idx, best_pos)
            n_jumped += 1
        end
    end

    D_grid = toroidal_dist_matrix(g, swarm.pos)
    return (n_jumped, s_fn(D_data, D_grid))
end

"""
    _linear_slope(ys) -> Float64

Slope of the least-squares line through ys, used to detect convergence.
When the slope of recent stress values drops below epsilon the radius is done.
"""
function _linear_slope(ys::Vector{Float64}) :: Float64
    n     = length(ys)
    xs    = Float64.(1:n)
    x̄     = mean(xs)
    ȳ     = mean(ys)
    num   = sum((xs .- x̄) .* (ys .- ȳ))
    denom = sum((xs .- x̄) .^ 2)
    denom == 0.0 && return 0.0
    return num / denom
end

"""
    swarm_radius!(swarm, D_data, g, pg, R; epsilon, max_epochs, window, rng) -> Vector{Float64}

Run epochs at a fixed radius R until convergence or max_epochs is hit.

Convergence is declared when the linear slope of the stress over the last
`window` epochs falls below `epsilon` — meaning the bots have stopped finding
better positions at this radius.  Returns the per-epoch stress history.
"""
function swarm_radius!(swarm::Swarm,
                       D_data::Matrix{Float64},
                       g::Grid,
                       pg::PolarGrid,
                       R::Int;
                       epsilon::Float64 = 0.01,
                       max_epochs::Int  = 200,
                       window::Int      = 10,
                       rng              = Random.default_rng(),
                       s_fn             = (dd,dg)->stress(dd,dg)) :: Vector{Float64}
    history = Float64[]
    for epoch in 1:max_epochs
        n_jumped, s = swarm_epoch!(swarm, D_data, g, pg, R; rng, s_fn)
        push!(history, s)

        # we evaluate after running at least 'window' epochs
        if length(history) >= window
            slope = _linear_slope(history[end-window+1:end])
            abs(slope) < epsilon && break
        end
    end
    return history
end

# ── Step 7: Driver ────────────────────────────────────────────────

"""
    PswarmResult

Everything the swarm left behind after training.

  positions      — final grid position of each bot (complex: real=row, imag=col)
  stress_history — stress after every epoch, across all radii
  radius_history — which R was active during each epoch
  grid           — the Grid the bots lived on
  polar          — the PolarGrid (ring offsets for each radius)
  final_stress   — stress of the converged placement
  D_data         — the original pairwise data distances, kept for show_grid and umatrix
"""
struct PswarmResult
    positions      :: Vector{ComplexF64}
    stress_history :: Vector{Float64}
    radius_history :: Vector{Int}
    grid           :: Grid
    polar          :: PolarGrid
    final_stress   :: Float64
    D_data         :: Matrix{Float64}
end

"""
    pswarm!(D_data; seed, epsilon, max_epochs, verbose) -> PswarmResult

Run the full TopoSwarm on a pairwise distance matrix.

Sets up a grid sized for the number of bots, places them randomly, then steps
through radii from Rmax down to Rmin.  At each radius, bots run epochs until the
stress slope flattens (< epsilon) or max_epochs is exhausted.

D_data must be a square, symmetric Float64 matrix with zeros on the diagonal —
the only input the swarm ever sees.
"""
function pswarm!(D_data::Matrix{Float64};
                 pop_weights::Union{Vector{Float64}, Nothing} = nothing,
                 seed::Union{Int,Nothing} = nothing,
                 epsilon::Float64         = 0.01,
                 max_epochs::Int          = 200,
                 verbose::Bool            = true) :: PswarmResult

    n = size(D_data, 1)
    size(D_data, 2) == n || error("D_data must be square")
    if !isnothing(pop_weights)
        length(pop_weights) == n || error("pop_weights length must equal n bots")
    end

    g   = make_grid(n)
    pg  = setup_polar_grid(g)
    rng = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)

    if verbose
        println("Bots  : $n")
        println("Grid  : $(g.lines) × $(g.cols) = $(g.lines*g.cols) cells")
        println("Rmax  : $(pg.rmax)  Rmin : $(pg.rmin)")
        println()
    end

    swarm      = init_swarm(g, n; seed)
    all_stress = Float64[]
    all_radii  = Int[]

    s_fn = isnothing(pop_weights) ?
        (dd, dg) -> stress(dd, dg) :
        (dd, dg) -> stress_weighted(dd, dg, pop_weights)

    for R in pg.rmax:-1:pg.rmin
        history = swarm_radius!(swarm, D_data, g, pg, R;
                                epsilon, max_epochs, rng, s_fn)
        append!(all_stress, history)
        append!(all_radii,  fill(R, length(history)))
        verbose && @printf("R=%2d  epochs=%3d  stress=%.4f\n",
                           R, length(history), history[end])
    end

    final_s = s_fn(D_data, toroidal_dist_matrix(g, swarm.pos))
    verbose && println("\nDone. Final stress : $(round(final_s, digits=4))")

    return PswarmResult(
        copy(swarm.pos),
        all_stress,
        all_radii,
        g, pg,
        final_s,
        D_data
    )
end

# ── Step 8: Visualisation ─────────────────────────────────────────

"""
    projection_data(result) -> (rows, cols)

Extract the final bot positions as separate row and column index vectors.
Useful for handing the projection off to an external plotting library.
"""
function projection_data(result::PswarmResult)
    rows = Int.(real.(result.positions))
    cols = Int.(imag.(result.positions))
    return rows, cols
end

"""
    umatrix(result) -> Matrix{Float64}

U-matrix: for each occupied cell, the average D_data distance to its directly
adjacent (cardinal) bot-occupied neighbours.

High values mark cluster boundaries; low values sit inside a cluster.
Empty cells are NaN.  Use show_grid for a combined projection + shade view
that fills empty cells with a neighbourhood-diversity score.
"""
function umatrix(result::PswarmResult) :: Matrix{Float64}
    g           = result.grid
    U           = fill(NaN, g.lines, g.cols)
    pos         = result.positions
    D_data      = result.D_data
    cell_to_bot = Dict(p => i for (i, p) in enumerate(pos))
    neighbours  = [ComplexF64(-1,0), ComplexF64(1,0),
                   ComplexF64(0,-1), ComplexF64(0,1)]

    for (i, p) in enumerate(pos)
        r, c  = real(p), imag(p)
        dists = Float64[]
        for offset in neighbours
            nr = mod(r - 1 + real(offset), g.lines) + 1
            nc = mod(c - 1 + imag(offset), g.cols)  + 1
            nb = ComplexF64(nr, nc)
            haskey(cell_to_bot, nb) &&
                push!(dists, D_data[i, cell_to_bot[nb]])
        end
        isempty(dists) || (U[Int(r), Int(c)] = mean(dists))
    end
    return U
end

# ── Step 9: ASCII Visualisation ───────────────────────────────────

const _SHADE_CHARS = [' ', '·', '░', '▒', '▓', '█']

"""
    _shade_char(val, lo, hi) -> Char

Map a scalar value in [lo, hi] to one of the six shade characters ' ' · ░ ▒ ▓ █.
NaN returns a space (no data); lo ≈ hi returns '·' (flat range, pick the midpoint).
"""
function _shade_char(val::Float64, lo::Float64, hi::Float64) :: Char
    isnan(val) && return ' '
    hi ≈ lo    && return '·'
    t = clamp((val - lo) / (hi - lo), 0.0, 1.0)
    return _SHADE_CHARS[round(Int, t * (length(_SHADE_CHARS) - 1)) + 1]
end

"""
    show_grid(result; labels, title) -> Nothing

Print the grid combining bot positions (shown by label) with a
neighbourhood-diversity shade for every empty cell.

Each occupied cell shows its bot label.
Each empty cell shows a shade character encoding the average pairwise
D_data distance of all bots within a small toroidal radius:

  ' ' · ░ ▒ ▓ █
  low ←————————→ high data distance  (darker = cluster boundary)

A cell surrounded only by same-cluster bots has low average pairwise
distance -> light shade.  A cell between two clusters has bots from both
sides -> high average distance -> dark shade.
"""
function show_grid(result::PswarmResult;
                   labels::Union{Vector{String},Nothing} = nothing,
                   title::String = "Grid") :: Nothing
    g   = result.grid
    n   = length(result.positions)
    lbs = isnothing(labels) ? string.(1:n) : labels
    D   = result.D_data
    R   = max(2, g.lines ÷ 8)   # neighbourhood radius scales with grid size

    # neighbourhood score: avg pairwise D_data of bots within R steps
    scores = fill(NaN, g.lines, g.cols)
    for r in 1:g.lines, c in 1:g.cols
        nearby = Int[]
        for (i, p) in enumerate(result.positions)
            dr = abs(Int(real(p)) - r)
            dc = abs(Int(imag(p)) - c)
            dr = min(dr, g.lines - dr)
            dc = min(dc, g.cols  - dc)
            sqrt(dr^2 + dc^2) <= R && push!(nearby, i)
        end
        length(nearby) < 2 && continue
        total, npairs = 0.0, 0
        for a in 1:length(nearby), b in (a+1):length(nearby)
            total  += D[nearby[a], nearby[b]]
            npairs += 1
        end
        scores[r, c] = total / npairs
    end

    vals = filter(!isnan, vec(scores))
    lo   = isempty(vals) ? 0.0 : minimum(vals)
    hi   = isempty(vals) ? 1.0 : maximum(vals)
    W    = max(2, maximum(length.(lbs))) + 1
    cell = Dict(result.positions[i] => lbs[i] for i in 1:n)

    println("$title  $(g.lines)×$(g.cols)  (label=bot  ░▒▓█=cluster boundary)\n")
    for r in 1:g.lines
        for c in 1:g.cols
            p  = ComplexF64(r, c)
            lb = get(cell, p, nothing)
            if lb !== nothing
                print(lpad(lb, W - 1), " ")
            else
                ch = _shade_char(scores[r, c], lo, hi)
                print(lpad(string(ch), W - 1), " ")
            end
        end
        println()
    end
    println()
    return nothing
end

"""
    show_stress(history; width, height, title) -> Nothing

Print an ASCII line chart of stress over epochs.

  hi ┤•
     │ •
     │  •
     │   •
  lo │    ••
     └──────
     1      n
"""
function show_stress(history::Vector{Float64};
                     width::Int   = 52,
                     height::Int  = 8,
                     title::String = "Stress History") :: Nothing
    isempty(history) && return nothing
    n  = length(history)
    lo = minimum(history)
    hi = maximum(history)
    hi ≈ lo && (hi += 1e-6)

    buf = fill(' ', height, width)
    for r in 1:height;    buf[r, 1]      = '│'; end
    for c in 2:width;     buf[height, c] = '─'; end
    buf[height, 1] = '└'

    for (i, v) in enumerate(history)
        c = 2 + round(Int, (i - 1) / max(n - 1, 1) * (width - 2))
        r = (height - 1) - round(Int, (v - lo) / (hi - lo) * (height - 2))
        buf[clamp(r, 1, height - 1), clamp(c, 2, width)] = '•'
    end

    println("$title  ($n epochs)\n")
    for r in 1:height
        lbl = r == 1          ? @sprintf("%.4f", hi) :
              r == height - 1  ? @sprintf("%.4f", lo) :
                                 "      "
        println(lbl, "  ", String(buf[r, :]))
    end
    @printf("        %s%s\n\n", "1", lpad(string(n), width - 1))
    return nothing
end

# ── Step 10: Live driver ──────────────────────────────────────────

"""
    pswarm_live!(D_data; seed, epsilon, max_epochs, fps, labels)

Same as pswarm! but redraws the grid and stress chart after every epoch so you
can watch the bots self-organise in real time.

In a Jupyter notebook the cell output is replaced each frame via
IJulia.clear_output; in a plain terminal ANSI escape codes clear the screen.
fps controls the target frame rate (default 4).  Returns a PswarmResult.
"""
function pswarm_live!(D_data::Matrix{Float64};
                      seed        = nothing,
                      epsilon     = 0.01,
                      max_epochs  = 200,
                      fps         = 4,
                      labels      = nothing)

    n      = size(D_data, 1)
    g      = make_grid(n)
    pg     = setup_polar_grid(g)
    rng    = isnothing(seed) ? Random.default_rng() : Random.MersenneTwister(seed)
    swarm  = init_swarm(g, n; seed)

    all_stress = Float64[]
    all_radii  = Int[]
    bot_labels = isnothing(labels) ? string.(1:n) : labels

    for R in pg.rmax:-1:pg.rmin
        history = Float64[]

        for epoch in 1:max_epochs
            n_jumped, s = swarm_epoch!(swarm, D_data, g, pg, R; rng)
            push!(history,    s)
            push!(all_stress, s)
            push!(all_radii,  R)

            tmp = PswarmResult(copy(swarm.pos), all_stress, all_radii, g, pg, s, D_data)

            # clear output: works in Jupyter via IJulia, falls back to ANSI in terminal
            if isdefined(Main, :IJulia)
                Main.IJulia.clear_output(true)
            else
                print("\033[2J\033[H")
            end

            show_grid(tmp; labels = bot_labels,
                      title = "R=$R  epoch=$epoch  stress=$(round(s,digits=4))  jumps=$n_jumped")
            show_stress(all_stress)
            sleep(1 / fps)

            if length(history) >= 10
                slope = _linear_slope(history[end-9:end])
                abs(slope) < epsilon && break
            end
        end
    end

    final_s = stress(D_data, toroidal_dist_matrix(g, swarm.pos))
    @printf("\nDone. Final stress: %.4f\n", final_s)
    return PswarmResult(copy(swarm.pos), all_stress, all_radii, g, pg, final_s, D_data)
end

# ── Step 11: Population-weighted stress ──────────────────────────

"""
    stress_weighted(D_data, D_grid, pop_weights) -> Float64

Population-weighted stress. Each pair (i,j) is weighted by the product of
their population counts, so nodes representing many raw samples dominate the
objective over sparse boundary nodes.

`pop_weights` should be normalised to sum to 1 (e.g. hit_map ./ sum(hit_map)).
Falls back to uniform weighting when all weights are equal.
"""
function stress_weighted(D_data      :: Matrix{Float64},
                         D_grid      :: Matrix{Float64},
                         pop_weights :: Vector{Float64}) :: Float64
    size(D_data) == size(D_grid) || error("distance matrices must have the same size")
    length(pop_weights) == size(D_data, 1) || error("pop_weights length must equal n nodes")
    nd      = normalise_matrix(D_data)
    ng      = normalise_matrix(D_grid)
    n       = size(nd, 1)
    total   = 0.0
    total_w = 0.0
    for i in 1:n
        for j in (i+1):n
            w        = pop_weights[i] * pop_weights[j]
            diff     = nd[i,j] - ng[i,j]
            total   += w * diff^2
            total_w += w
        end
    end
    return total / total_w
end

# ── Step 12: Raw sample assignment ───────────────────────────────

"""
    assign_raw_samples(result, bmu_indices) -> (rows, cols)

Map each raw sample to the grid position of its representative node.

`bmu_indices` is a vector of length n_raw_samples where each entry is the
flat node index (0-based, row-major: node_row * grid_cols + node_col) as
written by ESOM.export_npz.

Returns two Int vectors of length n_raw_samples: grid row and column per sample.
"""
function assign_raw_samples(result      :: PswarmResult,
                            bmu_indices :: Vector{Int}) :: Tuple{Vector{Int}, Vector{Int}}
    node_rows   = Int.(real.(result.positions))
    node_cols   = Int.(imag.(result.positions))
    sample_rows = node_rows[bmu_indices .+ 1]   # +1: Julia is 1-based, Python is 0-based
    sample_cols = node_cols[bmu_indices .+ 1]
    return sample_rows, sample_cols
end

# ── Step 13: Interpolated U-matrix ───────────────────────────────

"""
    umatrix_interpolated(result; R) -> Matrix{Float64}

Dense U-matrix: fills NaN cells from `umatrix()` by inverse-distance weighted
interpolation from occupied neighbours within radius R (toroidal).

Produces a full-grid matrix suitable for U*F flood-fill segmentation.
"""
function umatrix_interpolated(result :: PswarmResult; R :: Int = 3) :: Matrix{Float64}
    U_sparse = umatrix(result)
    U_dense  = copy(U_sparse)
    g        = result.grid

    for r in 1:g.lines, c in 1:g.cols
        isnan(U_sparse[r, c]) || continue
        total_w = 0.0
        total_v = 0.0
        for dr in -R:R, dc in -R:R
            nr = mod(r - 1 + dr, g.lines) + 1
            nc = mod(c - 1 + dc, g.cols)  + 1
            isnan(U_sparse[nr, nc]) && continue
            d = sqrt(Float64(dr^2 + dc^2))
            d == 0.0 && continue
            w        = 1.0 / d
            total_w += w
            total_v += w * U_sparse[nr, nc]
        end
        total_w > 0.0 && (U_dense[r, c] = total_v / total_w)
    end
    return U_dense
end

end # module TopoSwarm
