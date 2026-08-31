# StochasticOrders.jl

The stochastic orders, computed on samples.

> **Not** [StochasticDominance.jl](https://github.com/rajmadan96/StochasticDominance.jl),
> which is a portfolio-optimization package: it finds asset allocations subject
> to higher-order dominance constraints, and does `n`-th order dominance, which
> this package does not. This one is about the *partial order over a family* —
> the frontier, the antichain, the lattice, and the margins by which dominance
> fails — with no optimizer and no dependency beyond `Statistics`. Two equal-length vectors of draws
in; a verdict, a margin, or a frontier out.

```julia
using StochasticOrders

fosd(xs, ys)        # first-order: better for every increasing utility
sosd(xs, ys)        # second-order: better for every risk-averse preference
icx(xs, ys)         # increasing convex: the risk-seeking counterpart
statewise(xs, ys)   # better draw by draw, if the inputs are index-aligned
pgeq(xs, ys)        # P(x ≥ y) under that alignment

undominated(vs)     # indices nothing dominates — the maximal antichain
dominating(vs)      # the greatest element, or `nothing`
robust(vs)          # undominated across a sweep of risk aversions
margin(xs, ys)      # why it failed, and by how much
```

No dependency beyond `Statistics`. Works on anything you have draws of: an
MCMC chain's columns, a bootstrap, a CSV column, `MonteCarloMeasurements`
particles.

## Why it is cheap

For two equal-weight empirical measures, `F̂ₓ ≤ F̂ᵥ` pointwise iff the quantile
functions reverse, and both are step functions with jumps at the *same* `i/N`.
So the uncountable family of constraints collapses to exactly `N`, and the test
is an order-statistic comparison — no grid, no KDE, no binning:

```julia
all(sort(xs) .>= sort(ys))
```

`O(N log N)`, and **exact on the empirical measures**: the predicate computed is
precisely the predicate defined. Everything else is that same sorted difference
under a different reduction. With `d = sort(xs) .- sort(ys)`:

| reduction | meaning |
|---|---|
| `minimum(d) >= 0` | first-order dominance |
| `mean(max.(-d, 0))` | `V₁`, the one-sided W₁ violation; zero iff dominance holds |
| `mean(abs.(d))` | Wasserstein-1 |
| `mean(d)` | `E[x] - E[y]`, and equal to W₁ when dominance holds |
| `max.(sort, sort)` | the lattice join, `⊔` |
| `cumsum(sort(·))` | second-order dominance |

Equal length is what makes the collapse work. With `N ≠ M` you would have to
compare on the union of breakpoints and the index-wise test is simply wrong, so
every entry point checks.

## Read it as a decision rule, not a test

There is no size and no power. The predicate is a conjunction of `N`
inequalities, so it is heavily biased *against* declaring dominance, and its
two most fragile constraints — `i = 1` and `i = N` — are the noisiest
quantities in the sample. For a confidence statement you want a
Barrett–Donald-style bootstrap, which this package does not do.

What rescues it in practice is **coupling**. If the vectors are index-aligned
draws of the same underlying randomness — common random numbers, a shared
chain, a library that propagates correlations — their sorted versions are
positively dependent and the test sharpens enormously. `xs = ys .+ δ` passes at
a single draw; two independent samples with the same marginals need `δ` to clear
the extreme-order-statistic noise.

## Expect nothing to dominate

First-order dominance requires the extreme order statistics to be ordered, and
between two location-scale distributions it requires *equal scale*. It is
essentially a same-shape-shifted relation, so any two options differing in
spread are incomparable after two comparisons. In a decision whose whole point
is that the options differ in risk, that is nearly every pair.

`undominated` therefore returns nearly everything — and needs no sentinel, since
"nothing dominates anything" and "everything is undominated" are the same
statement. `dominating` is where `nothing` belongs, as with `findfirst`.

Second-order dominance bites more often but has its own asymmetry: its `k = N`
constraint is `E[x] ≥ E[y]`, and the risky option usually has the higher mean,
so the safe one fails on the mean while the risky one fails on the left tail.

So the functions that actually discriminate are `margin`, `regret`, and
`robust`. A margin localises the failure — "fails at 2.6% of quantile levels,
all above p = 0.97, worst 0.41, `P(x ≥ y) = 0.94`" is a decision; `false` is
not. `robust` sweeps exponential-utility risk aversion and keeps what survives
every setting, which is the claim a decision-maker can act on.

## The lattice

`⊔` and `⊓` are the join and meet — the pointwise minimum of two cdfs is a cdf,
and cdf-min is quantile-max, so the join is `max.(sort(xs), sort(ys))`. `⊔` sits
in Julia's precedence group 11 (with `+`), `⊓` in group 12 (with `*`); neither
is in Base.

A finite family forms a *poset* under dominance, not a lattice — the join is
generally none of its members. `envelope(vs)` materialises it anyway as an
unattainable best case, which is what `regret` measures distance to.

## Tests

```fish
julia --project=. -e 'using Pkg; Pkg.test()'
```
