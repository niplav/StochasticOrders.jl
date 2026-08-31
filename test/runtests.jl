using StochasticOrders
using Test

@testset "StochasticOrders" begin


@testset "first order" begin
	a, b = collect(1.0:100), collect(0.0:99)
	@test fosd(a, b)
	@test !fosd(b, a)
	@test fosd(a, a)                       # weak: equality counts
	@test relate(a, b) === :dominates
	@test relate(b, a) === :dominated
	@test relate(a, a) === :equal
	# a crossing: same mean, different spread
	c = [-10.0, 0, 0, 10]
	d = [-1.0, 0, 0, 1]
	@test !fosd(c, d) && !fosd(d, c)
	@test relate(c, d) === :crossing
	@test_throws DimensionMismatch fosd([1.0, 2], [1.0, 2, 3])
end

@testset "second order and convex" begin
	c = [-10.0, 0, 0, 10]      # mean-preserving spread of d
	d = [-1.0, 0, 0, 1]
	@test sosd(d, c)           # risk-averse prefers the tight one
	@test !sosd(c, d)
	@test icx(c, d)            # risk-seeking prefers the spread
	@test !icx(d, c)
	# a mean-preserving spread must not fail its own order on rounding
	@test icx(2 .* d, d) && sosd(d, 2 .* d)
	@test fosd(collect(1.0:10), collect(0.0:9))
	@test sosd(collect(1.0:10), collect(0.0:9))    # first order implies second
end

@testset "statewise and pgeq" begin
	a = [3.0, 1, 2]
	@test statewise(a .+ 1, a)
	@test pgeq(a .+ 1, a) == 1.0
	@test pgeq(a, a) == 1.0                # >= is weak
	# sorted-dominates but not statewise
	x, y = [0.0, 3.0], [1.0, 2.0]
	@test fosd([0.0, 3.0], [0.0, 2.0])
	@test !statewise(x, y)
	@test pgeq(x, y) == 0.5
    # statewise implies fosd
	@test fosd(a .+ 1, a)
end

@testset "margin" begin
	a, b = collect(1.0:100), collect(0.0:99)
	m = margin(a, b)
	@test m.dominates && m.v1 == 0
	@test m.meangap ≈ 1
	@test occursin("dominates", sprint(show, m))
	m2 = margin([-10.0, 0, 0, 10], [-1.0, 0, 0, 1])
	@test !m2.dominates && m2.v1 > 0
	@test m2.frac ≈ 0.25                   # one of four levels violated
	@test m2.atlevel ≈ 0.125               # and it is the lowest one
	@test occursin("crosses", sprint(show, m2))
	@test wass(a, b) ≈ 1
	@test wass(a, b) ≈ m.meangap           # equal under dominance
end

@testset "lattice" begin
	x, y = [0.0, 5.0, 2.0], [1.0, 1.0, 9.0]
	j, mt = x ⊔ y, x ⊓ y
	@test j == [1.0, 2.0, 9.0] && mt == [0.0, 1.0, 5.0]
	@test fosd(j, x) && fosd(j, y)
	@test fosd(x, mt) && fosd(y, mt)
	@test lub(x, y) == j && glb(x, y) == mt
	e = envelope([x, y, [3.0, 3.0, 3.0]])
	@test all(fosd(e, v) for v in (x, y, [3.0, 3.0, 3.0]))
	r = regret([x, y])
	@test all(>=(0), r)
	@test regret([x, x]) == [0.0, 0.0]      # the join of one thing is itself
end

@testset "collections" begin
	chain = [collect(float(i):(i + 9)) for i in 0:4]     # a total order
	@test undominated(chain) == [5]
	@test dominating(chain) == 5
	@test domlevels(chain) == [4, 3, 2, 1, 0]
	@test sum(hasse(chain)) == 4
	@test sum(domgraph(chain)) == 10                     # transitive closure
	@test all(iszero, [margins(chain)[5, j] for j in 1:4])

	# mean and spread both rise: nothing is comparable
	spread = [[-(1.0 + i), float(i), 1.0 + 2i] for i in 0:4]
	@test undominated(spread) == collect(1:5)
	@test dominating(spread) === nothing
	@test all(iszero, domlevels(spread))

	@test length(robust(spread)) >= 1
	@test robust(chain) == [5]                # a total order has one survivor
	@test_throws DimensionMismatch undominated([[1.0, 2], [1.0, 2, 3]])
end

@testset "no hidden length assumptions" begin
	for n in (1, 2, 3, 17, 100)
		a = collect(1.0:n)
		@test fosd(a, a) && relate(a, a; order=:first) === :equal
		@test margin(a, a).v1 == 0
		@test undominated([a, a]) == [1, 2]   # identical: neither dominates
	end
end

end
