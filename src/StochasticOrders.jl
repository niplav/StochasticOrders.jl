"""
StochasticOrders — the stochastic orders, computed on samples.

Given two equal-length vectors of draws, decide whether one distribution
dominates the other, by how much it fails when it does not, and which members
of a family nothing dominates.

	using StochasticOrders
	fosd(xs, ys)          # first-order
	undominated(vs)       # indices nothing dominates
	margin(xs, ys)        # why it failed, and by how much

Everything here is one sorted vector away from everything else. With
`d = sort(xs) .- sort(ys)` and equal lengths:

	minimum(d) >= 0		first-order dominance, exact on the empirical measures
	mean(max.(-d, 0))	the one-sided W₁ violation; zero iff dominance holds
	mean(abs.(d))		Wasserstein-1 distance
	mean(d)			E[X] - E[Y], and equal to W₁ when dominance holds
	max.(sort, sort)	the lattice join;  min.(sort, sort) the meet

Why the sorted vectors are the whole story: `F̂ₓ ≤ F̂ᵥ` pointwise iff the
quantile functions satisfy `Q̂ₓ ≥ Q̂ᵥ`, and for two equal-weight N-atom measures
both quantile functions are step functions with jumps at the *same* `p = i/N`.
So the uncountable family of constraints collapses to exactly N. **Equal length
is what makes it collapse** — with N ≠ M you would have to compare on the union
of breakpoints, and the index-wise test would be wrong. Every entry point here
checks it.

Read the verdicts as a decision rule, not a hypothesis test. There is no size
and no power: the predicate is a conjunction of N inequalities, so it is heavily
biased *against* declaring dominance, and its two most fragile constraints
(`i = 1` and `i = N`) are the noisiest quantities in the sample.

What rescues it in practice is coupling. If the two vectors are index-aligned —
draws of the same underlying randomness, as produced by common random numbers,
a shared MCMC chain, or a Monte Carlo library that propagates correlations —
then their sorted versions are positively dependent and the test becomes far
sharper. `xs = ys .+ δ` passes at a single draw; two independent samples with
the same marginals need δ to clear the noise in the extreme order statistics.
Alignment also makes [`statewise`](@ref) and [`pgeq`](@ref) meaningful at all.
"""
module StochasticOrders

using Statistics: mean, quantile

export relate, fosd, sosd, icx, statewise, pgeq, wass,
	margin, DomMargin,
	lub, glb, ⊔, ⊓, envelope, regret,
	domgraph, hasse, domlevels, undominated, dominating, margins, robust

const Vec = AbstractVector{<:Real}

# --------------------------------------------------------------- the orders

"""
	relate(xs, ys; order=:first)

How `xs` and `ys` stand to each other: `:dominates`, `:dominated`, `:equal`, or
`:crossing`. One traversal decides both directions.

Orders: `:first` (FOSD), `:second` (increasing concave utilities, i.e. every
risk-averse preference), `:convex` (increasing convex — risk-seeking),
`:statewise` (pathwise, requiring index-aligned inputs).
"""
relate(xs::Vec, ys::Vec; order::Symbol=:first) =
	_cmp(score(xs, order), score(ys, order))

"""
	score(xs, order)

The constraint vector whose componentwise ordering *is* the given stochastic
order. Sorted samples for `:first`, their cumulative sums for `:second`, upper
partial sums for `:convex`, and the raw index-aligned draws for `:statewise`.

`:second` and `:convex` integrate the cdf, and those integrals diverge for a
non-integrable tail — population second-order dominance is simply undefined for
a Cauchy, while these partial sums are always finite and will happily return a
verdict about a quantity that does not exist. Nothing observable in a sample
vector reveals that, so **the caller must rule it out.**
"""
function score(xs::Vec, order::Symbol)
	order === :statewise && return xs
	s = sort(xs)
	order === :first && return s
	order === :second && return cumsum(s)
	order === :convex && return reverse!(cumsum(reverse(s)))
	error("unknown order $order; want :first, :second, :convex or :statewise")
end

# Probe the discriminating indices first — the extremes, then a dyadic ladder —
# and bail the moment both directions are dead. That is a quantile pre-filter at
# every resolution at once, with no separate pass: since dominance implies every
# quantile dominates, an early mismatch in both directions is conclusive. A full
# sweep still follows, because *declaring* dominance requires checking all N.
const LADDER = (0.5, 0.999, 0.001, 0.25, 0.75, 0.1, 0.9, 0.02, 0.98, 0.35,
	0.65, 0.15, 0.85, 0.4, 0.6, 0.05, 0.95)

# Ties are decided within rounding, not exactly. This matters: for the
# cumulative orders the boundary constraint is an exact *equality* of means —
# `:second` at k = N, `:convex` at k = 1 — so two genuinely equal-mean options
# would otherwise be separated by whichever way `sum` happened to round, and a
# mean-preserving spread would fail the convex order it defines.
tol(a::Vec, b::Vec) = 1e-12 * max(maximum(abs, a), maximum(abs, b), 1.0)

function _cmp(a::Vec, b::Vec)
	n = length(a)
	n == length(b) || throw(DimensionMismatch(
		"need equal sample counts, got $n and $(length(b)); the " *
		"order-statistic equivalences do not hold otherwise"))
	t = tol(a, b)
	lo = hi = false
	@inbounds begin
		for i in (1, n)
			d = a[i] - b[i]
			d < -t && (lo = true)
			d > t && (hi = true)
			lo && hi && return :crossing
		end
		for p in LADDER
			i = clamp(1 + floor(Int, p * n), 1, n)
			d = a[i] - b[i]
			d < -t && (lo = true)
			d > t && (hi = true)
			lo && hi && return :crossing
		end
		for i in 1:n
			d = a[i] - b[i]
			d < -t && (lo = true)
			d > t && (hi = true)
			lo && hi && return :crossing
		end
	end
	lo ? :dominated : hi ? :dominates : :equal
end

"""
	fosd(xs, ys)

Does `xs` first-order stochastically dominate `ys` (weakly)? Equality counts.

Dominance is essentially a *same-shape, shifted* relation: it requires the
extreme order statistics to be ordered, and between two location-scale
distributions it requires equal scale. Two options that differ in spread are
incomparable after two element comparisons, however good one is elsewhere. In
a decision whose point is that options differ in risk, expect most pairs to
cross — that is information, not a malfunction.
"""
fosd(xs::Vec, ys::Vec) = relate(xs, ys; order=:first) in (:dominates, :equal)

"""
	sosd(xs, ys)

Second-order dominance: `E[u(x)] ≥ E[u(y)]` for every increasing *concave* `u`.
Equivalent to the lower partial sums of the order statistics dominating.

The `k = N` constraint is exactly `E[x] ≥ E[y]`, so it comes for free and must
not be imposed separately. It is also why this order bites less often than it
looks: in a cost-benefit setting the risky option usually has the higher mean,
so the safe option fails on the mean while the risky one fails on the left
tail, and the pair is incomparable again.

Requires an integrable tail — see [`score`](@ref).
"""
sosd(xs::Vec, ys::Vec) = relate(xs, ys; order=:second) in (:dominates, :equal)

"""
	icx(xs, ys)

Increasing convex order — the risk-*seeking* counterpart of [`sosd`](@ref),
`E[u(x)] ≥ E[u(y)]` for every increasing convex `u`. Upper partial sums.
"""
icx(xs::Vec, ys::Vec) = relate(xs, ys; order=:convex) in (:dominates, :equal)

"""
	statewise(xs, ys)
	pgeq(xs, ys)

`statewise` asks whether `xs ≥ ys` draw by draw with the index alignment
intact — better in *every state of the world*. Strictly stronger than
[`fosd`](@ref), and meaningful only for aligned inputs: for two independent
samples the pairing is an arbitrary permutation and the verdict is an artefact
of it.

By Strassen's theorem, first-order dominance holds iff *some* coupling makes
`x ≥ y` almost surely, and the comonotone (sort-both) coupling is the one that
achieves it. So these are the same predicate under two couplings: the one your
data actually has, and the canonical optimistic one.

Prefer `pgeq`, which returns `P(x ≥ y)` under the given alignment. A
probability-one claim is not estimable from N draws — a true 0.9995 reads as
`false` and tells you nothing — whereas `pgeq` is, at `O(N^-1/2)`. `statewise`
is its degenerate `== 1.0` case.
"""
statewise(xs::Vec, ys::Vec) = relate(xs, ys; order=:statewise) in (:dominates, :equal)

function pgeq(xs::Vec, ys::Vec)
	_samelen(xs, ys)
	count(i -> xs[i] >= ys[i], eachindex(xs)) / length(xs)
end

"""
	wass(xs, ys)

Wasserstein-1 distance. Equals `mean(xs) - mean(ys)` exactly when `xs`
dominates `ys`, since then the quantile difference never changes sign.
"""
function wass(xs::Vec, ys::Vec)
	_samelen(xs, ys)
	sum(abs, sort(xs) .- sort(ys)) / length(xs)
end

_samelen(a, b) = length(a) == length(b) || throw(DimensionMismatch(
	"need equal sample counts, got $(length(a)) and $(length(b))"))

# ---------------------------------------------------------------- the margin

"""
	DomMargin

Why dominance failed, and by how much. A bare `false` is not a decision; this
is. `v1` is zero exactly when dominance holds.

- `v1` — the one-sided W₁ violation, the area over which the cdfs are in the
  wrong order, equivalently the average quantile shortfall. A *sum*, so one
  unlucky tail draw cannot dominate it — unlike `minimum(d)`, which is a max
  over N noisy differences.
- `worst`, `atlevel` — the largest single violation and the quantile level it
  occurs at. The level is the actionable part: a failure confined above
  `p = 0.97` is a different story from one at the median.
- `frac` — the fraction of quantile levels violated.
- `meangap` — `E[x] - E[y]`. This is *also* the signed area between the cdfs,
  which is why signed area is useless as a dominance margin: a large positive
  value is perfectly consistent with gross crossing.
- `pgeq` — `P(x ≥ y)` under the given alignment.
"""
struct DomMargin
	dominates::Bool
	v1::Float64
	worst::Float64
	atlevel::Float64
	frac::Float64
	meangap::Float64
	pgeq::Float64
end

function margin(xs::Vec, ys::Vec; order::Symbol=:first)
	a, b = score(xs, order), score(ys, order)
	_samelen(a, b)
	n = length(a)
	t = tol(a, b)          # same tie rule as `_cmp`, so `v1 == 0` and the
	                       # predicate can never disagree
	v1 = 0.0; worst = 0.0; at = 0; bad = 0; tot = 0.0
	@inbounds for i in 1:n
		d = a[i] - b[i]
		tot += d
		if d < -t
			v1 -= d
			bad += 1
			-d > worst && (worst = -d; at = i)
		end
	end
	DomMargin(bad == 0, v1 / n, worst, at == 0 ? NaN : (at - 0.5) / n,
		bad / n, tot / n, pgeq(xs, ys))
end

function Base.show(io::IO, m::DomMargin)
	f(x) = string(round(x, sigdigits=4))
	if m.dominates
		print(io, "dominates (mean gap ", f(m.meangap),
			", P(x≥y) = ", round(m.pgeq, digits=3), ")")
	else
		print(io, "crosses: fails at ", round(100 * m.frac, digits=1),
			"% of quantile levels, worst ", f(m.worst), " at p = ",
			round(m.atlevel, digits=3), "; V₁ = ", f(m.v1),
			", mean gap ", f(m.meangap),
			", P(x≥y) = ", round(m.pgeq, digits=3))
	end
end

# --------------------------------------------------------------- the lattice

"""
	lub(xs, ys)   (also `xs ⊔ ys`)
	glb(xs, ys)   (also `xs ⊓ ys`)

Join and meet in the dominance lattice, as **sorted** sample vectors. The
pointwise minimum of two cdfs is a cdf, and cdf-min is quantile-max, so the
join is the elementwise maximum of the sorted samples.

`⊔` sits in Julia's precedence group 11 (with `+`) and `⊓` in group 12 (with
`*`), so `a ⊔ b ⊓ c` parses as `a ⊔ (b ⊓ c)`. Neither is defined in Base.

The result carries no alignment with its arguments — it is nobody's function of
them — so treat it as a fresh marginal, and shuffle it before pairing it with
anything.
"""
function lub(xs::Vec, ys::Vec)
	_samelen(xs, ys)
	max.(sort(xs), sort(ys))
end

function glb(xs::Vec, ys::Vec)
	_samelen(xs, ys)
	min.(sort(xs), sort(ys))
end

const ⊔ = lub
const ⊓ = glb

"""
	envelope(vs)

The least distribution dominating every member of `vs`, as a sorted vector —
`reduce(⊔, vs)` in one pass. An unattainable best case, and the natural
denominator for [`regret`](@ref).

Note that `vs` forms a *poset* under dominance, not a lattice: this join is
generally none of its members. The lattice is on all distributions over ℝ; a
finite family sits inside it and is a sublattice only by accident.
"""
function envelope(vs::AbstractVector{<:Vec})
	isempty(vs) && error("envelope of nothing")
	v = sort(first(vs))
	for x in Iterators.drop(vs, 1)
		_samelen(v, x)
		v = max.(v, sort(x))
	end
	v
end

"""
	regret(vs)

For each member, its Wasserstein-1 distance to `envelope(vs)` — how much of the
best-case quantile profile that option gives up. Usable when the frontier is
uninformative, which it usually is.
"""
function regret(vs::AbstractVector{<:Vec})
	e = envelope(vs)
	[sum(abs, e .- sort(x)) / length(e) for x in vs]
end

# ----------------------------------------------------------------- the order

"""
	domgraph(vs; order=:first)

`G × G` relation, `[i, j]` true when `i` *strictly* dominates `j`. Equality is
recorded neither way, which keeps the relation acyclic.

Sorted once up front into a column-major score matrix, so each comparison walks
two contiguous vectors. The worst case is unavoidably `O(G²)`: if the answer is
a size-`G` antichain, certifying it means looking at essentially every pair. So
the win lives in the constant, which the early exit in the comparison buys.
"""
function domgraph(vs::AbstractVector{<:Vec}; order::Symbol=:first)
	m = _scores(vs, order)
	g = length(vs)
	d = falses(g, g)
	mu = vec(sum(m, dims=1))
	for i in 1:g, j in 1:g
		# dominance implies the mean dominates, so this screen is sound and
		# skips both the pair and its mirror
		(i == j || mu[i] < mu[j]) && continue
		_cmp(view(m, :, i), view(m, :, j)) === :dominates && (d[i, j] = true)
	end
	d
end

function _scores(vs::AbstractVector{<:Vec}, order::Symbol)
	isempty(vs) && error("no candidates")
	cols = [score(x, order) for x in vs]
	n = length(first(cols))
	all(c -> length(c) == n, cols) || throw(DimensionMismatch(
		"candidates have different sample counts; the order-statistic " *
		"equivalences require a single count"))
	reduce(hcat, cols)
end

"""
	undominated(vs; order=:first)

Indices nothing strictly dominates — the maximal antichain.

Never empty, and needs no sentinel: "nothing dominates anything" and
"everything is undominated" are the same statement. Expect it to return nearly
all of `vs`.
"""
function undominated(vs::AbstractVector{<:Vec}; order::Symbol=:first)
	d = domgraph(vs; order)
	findall(j -> !any(@view d[:, j]), 1:length(vs))
end

"""
	dominating(vs; order=:first)

The greatest element — the one dominating all the others — or `nothing` when
there isn't one, as with `findfirst`. This is where a sentinel belongs;
[`undominated`](@ref) is where it does not.
"""
function dominating(vs::AbstractVector{<:Vec}; order::Symbol=:first)
	d = domgraph(vs; order)
	g = length(vs)
	findfirst(i -> all(j -> j == i || d[i, j], 1:g), 1:g)
end

"""
	domlevels(vs; order=:first)

Height in the dominance order: `0` for the undominated, `1` for those dominated
only by level 0, and so on. A coarse ranking that survives the frontier being
large.
"""
function domlevels(vs::AbstractVector{<:Vec}; order::Symbol=:first)
	d = domgraph(vs; order)
	g = length(vs)
	lv = fill(-1, g)
	for _ in 1:g
		changed = false
		for j in 1:g
			lv[j] >= 0 && continue
			ups = findall(@view d[:, j])
			if isempty(ups)
				lv[j] = 0; changed = true
			elseif all(i -> lv[i] >= 0, ups)
				lv[j] = 1 + maximum(lv[i] for i in ups); changed = true
			end
		end
		changed || break
	end
	lv
end

"""
	hasse(vs; order=:first)

Transitive reduction of [`domgraph`](@ref) — the cover relation, which is what
you want to draw. `O(G³)` bit operations; fine to a few hundred candidates.
"""
function hasse(vs::AbstractVector{<:Vec}; order::Symbol=:first)
	d = domgraph(vs; order)
	g = length(vs)
	h = copy(d)
	for i in 1:g, j in 1:g
		d[i, j] || continue
		any(k -> k != i && k != j && d[i, k] && d[k, j], 1:g) && (h[i, j] = false)
	end
	h
end

"""
	margins(vs; order=:first)

`G × G` of `V₁` violations: `[i, j]` is how badly `i` fails to dominate `j`,
and zero when it succeeds.
"""
function margins(vs::AbstractVector{<:Vec}; order::Symbol=:first)
	m = _scores(vs, order)
	g, n = length(vs), size(m, 1)
	out = zeros(g, g)
	for i in 1:g, j in 1:g
		i == j && continue
		v = 0.0
		@inbounds for k in 1:n
			d = m[k, i] - m[k, j]
			d < 0 && (v -= d)
		end
		out[i, j] = v / n
	end
	out
end

"""
	robust(vs; gammas=0:0.5:4)

Indices undominated under a sweep of risk aversions — the ranking that actually
discriminates when both [`fosd`](@ref) and [`sosd`](@ref) return large
antichains.

Uses exponential (CARA) certainty equivalents rather than CRRA, because
outcomes may be negative and CRRA is undefined there. The scale is the pooled
interquartile range, so `gamma` is dimensionless; `gamma = 0` is the plain mean.

An option surviving here is undominated for *every* risk aversion in the sweep,
which is a robustness claim a decision-maker can act on, and the induced order
is far coarser than second-order dominance because the family of utilities is
finite. The answer depends on the range swept — which is honest: it *is* the
question of which risk aversions you entertain.
"""
function robust(vs::AbstractVector{<:Vec}; gammas=0:0.5:4)
	pool = reduce(vcat, vs)
	scale = quantile(pool, 0.75) - quantile(pool, 0.25)
	scale > 0 || (scale = 1.0)
	mid = quantile(pool, 0.5)   # one shared centre, or the shift would not
	                            # cancel and the utilities would not compare
	gs = collect(float.(gammas))
	u = [_cara(x, g, scale, mid) for g in gs, x in vs]
	g = length(vs)
	findall(j -> !any(i -> i != j &&
		all(k -> u[k, i] >= u[k, j], axes(u, 1)) &&
		any(k -> u[k, i] > u[k, j], axes(u, 1)), 1:g), 1:g)
end

# Certainty equivalent under exponential utility, mid - log(E[exp(-a(x-mid))])/a.
# `mid` only keeps the exponential in range; it cancels because it is shared.
function _cara(xs::Vec, g::Real, scale::Real, mid::Real)
	iszero(g) && return sum(xs) / length(xs)
	a = g / scale
	mid - log(sum(x -> exp(-a * (x - mid)), xs) / length(xs)) / a
end

end # module
