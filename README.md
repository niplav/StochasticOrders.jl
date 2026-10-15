# StochasticOrders.jl

Stochastic dominance, computed on samples. Give it equal-length vectors of
draws — MCMC chain columns, bootstrap replicates, Monte Carlo output — and get
back a verdict, a margin by which dominance fails, or the undominated subset of
a family. The only dependency is `Statistics`.

(Not to be confused with
[StochasticDominance.jl](https://github.com/rajmadan96/StochasticDominance.jl),
which optimizes portfolios under dominance constraints.)

## Usage

```julia
using StochasticOrders

# pairs
fosd(xs, ys)        # first order: better for every increasing utility
sosd(xs, ys)        # second order: better for every risk-averse utility
icx(xs, ys)         # increasing convex: better for every risk-seeking utility
statewise(xs, ys)   # better draw by draw (index-aligned inputs only)
pgeq(xs, ys)        # P(x ≥ y) under the given alignment
relate(xs, ys)      # :dominates, :dominated, :equal or :crossing
margin(xs, ys)      # where and by how much dominance fails

# families (a vector of sample vectors)
undominated(vs)     # indices nothing dominates
dominating(vs)      # index of the greatest element, or `nothing`
robust(vs)          # undominated across a sweep of risk aversions
regret(vs)          # each option's W₁ distance to the best-case envelope
domgraph(vs)        # dominance relation as a G×G matrix; `hasse` reduces it
```

`relate`, `margin` and the family functions other than `robust` and `regret`
take `order=:first` (default), `:second`, `:convex` or `:statewise`.

## How it works

For two equal-size empirical distributions, first-order dominance holds iff
the sorted samples are ordered elementwise:

```julia
all(sort(xs) .>= sort(ys))
```

This is exact on the empirical measures, `O(N log N)`, with no grids or
density estimation. Second order uses cumulative sums of the sorted samples,
and the convex order uses upper partial sums. The margin, Wasserstein distance
and lattice operations (`⊔`, `⊓`, `envelope`) are all reductions of the same
sorted difference.

This only works with **equal sample counts**, so every function checks it.

## Caveats

**It's a decision rule, not a hypothesis test.** There's no size or power.
Dominance is a conjunction of `N` inequalities, the most fragile of which
involve the sample extremes, so it's biased against declaring dominance. For
confidence statements, use a bootstrap test (e.g. Barrett–Donald).

**Coupled samples help a lot.** If inputs are index-aligned draws of the same
randomness (common random numbers, a shared chain), the test is far sharper
than on independent samples. Alignment is also what makes `statewise` and
`pgeq` meaningful at all.

**Expect most pairs to be incomparable.** First-order dominance between two
location-scale distributions requires equal scale, so options that differ in
risk usually cross. Second order fails often too, since the riskier option
tends to have the higher mean. Hence `undominated` typically returns nearly
everything, and the informative functions are `margin`, `regret` and `robust`.

**Second and convex orders need integrable tails.** On samples from e.g. a
Cauchy they will return a verdict about a quantity that doesn't exist.

## Tests

```fish
julia --project=. -e 'using Pkg; Pkg.test()'
```
