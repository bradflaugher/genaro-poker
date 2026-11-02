// Genaro's teacher: a decision-time search solver on PokerSI's own engine.
//
// For each legal move (several raise sizes, every discard choice) it estimates the move's
// expected chips by Monte Carlo rollouts:
//   1. sample a world: keep every card the acting seat can see where it is in the deck,
//      reshuffle the rest (opponents' hidden cards, the cards still to come);
//   2. replay the hand so far in that world. Each opponent's recorded action is scored by how
//      likely that opponent would have taken it holding the sampled cards (Bayesian range
//      inference), so a world where the raiser holds junk counts for little;
//   3. make the candidate move and play the hand out, every seat by a House Bot with its
//      own style (common random numbers: every candidate sees the same worlds and bots).
// This is the rollout / policy-improvement scheme behind search-based multiway poker bots
// (as in Pluribus-style depth-limited search): one step of search on top of a base policy
// improves on that policy. It is not an equilibrium (CFR) solver: it best-responds to
// House-Bot-like opponents, which is what PokerSI seats.
//
// The persona keeps its flavour: among moves whose EV is close, its style (aggression,
// tightness, bluffiness) breaks the tie, and a bluffer may take a slightly worse bluff.
package genaro

import com.bradflaugher.pokersi.engine.*
import kotlin.math.max
import kotlin.math.roundToInt
import kotlin.random.Random

/** Everything needed to rebuild a hand from scratch and replay it. */
class HandRecord(
    val variant: GameVariant, val names: List<String>, val chips: List<Int>, val button: Int,
    val stakes: Stakes, val seed: Long, val handNumber: Int,
) {
    val moves = mutableListOf<Pair<Int, Decision>>()

    fun start(deck: Deck? = null) = Hand(variant, names.indices.map { SeatSetup(names[it], chips[it]) }, button, stakes, seed, handNumber, deck)

    /** The real deck order (Hand shuffles fullDeck with Random(seed)). */
    fun deckOrder(): List<Card> = Deck.fullDeck(variant.usesShortDeck).shuffled(Random(seed))
}

data class SolverParams(
    val worlds: Int = 480,
    /** Equity samples per House Bot decision inside rollouts and range inference. */
    val botIterations: Int = 40,
    /** Weight of a world in which an opponent's recorded action is not what it would do. */
    val mismatch: Double = 0.25,
    val styleBonusBb: Double = 0.35,
    /** Moves are ranked by EV minus this many standard errors of their edge over the passive move. */
    val confidence: Double = 1.0,
    /** Chance an opponent keeps its own House Bot style in a rollout (else a population style). */
    val ownStyle: Double = 0.5,
)

data class SolverResult(val choice: Decision, val evBb: Map<Decision, Double>, val seBb: Map<Decision, Double>, val worlds: Int, val effectiveWorlds: Double)

object Solver {
    /** Rollout stand-ins for opponents who don't fold like House Bot (LLMs, the human). */
    private val POPULATION = listOf(
        BotStyle("Sticky caller", tightness = 0.1, aggression = 0.3, bluffing = 0.05),
        BotStyle("Sound regular", tightness = 0.4, aggression = 0.55, bluffing = 0.1),
    )

    private fun category(d: Decision): Int = when (d) {
        Decision.Fold -> 0
        Decision.Check, Decision.Call -> 1
        is Decision.RaiseTo, Decision.AllIn -> 2
        is Decision.Discard -> 3
    }

    fun candidates(ctx: DecisionContext): List<Decision> {
        val l = ctx.legal
        l.discard?.let { rule ->
            val n = ctx.holeCards.size
            val out = mutableListOf<Decision>()
            for (mask in 0 until (1 shl n)) {
                val k = Integer.bitCount(mask)
                if (k in rule.min..rule.max) out += Decision.Discard((0 until n).filter { mask shr it and 1 == 1 }.toSet())
            }
            return out
        }
        val out = mutableListOf<Decision>()
        if (l.canCheck) out += Decision.Check else { out += Decision.Fold; out += Decision.Call }
        val lo = l.minRaiseTo
        val hi = l.maxRaiseTo
        if (lo != null && hi != null) {
            val me = ctx.seats.first { it.seat == ctx.seat }
            val potAfterCall = ctx.pot + l.toCall
            val sizes = listOf(0.0, 0.5, 1.0).map { f -> (me.betThisRound + l.toCall + f * potAfterCall).roundToInt().coerceIn(lo, hi) } + lo
            // Bet-size abstraction, as real solvers use: the all-in is a candidate only when it
            // isn't a giant overbet (the raise is at most 1.5 pots) or the stack is short.
            val jam = hi - (me.betThisRound + l.toCall)
            val withJam = if (jam <= 1.5 * potAfterCall || hi <= 25 * ctx.bigBlind) sizes + hi else sizes
            withJam.distinct().sorted().forEach { out += Decision.RaiseTo(it) }
        }
        return out
    }

    /** A world: the hand replayed with hidden cards resampled, and how plausible it is. */
    private class World(val order: List<Card>, val weight: Double)

    private fun sampleWorld(rec: HandRecord, hero: Int, known: Set<Card>, styles: List<BotStyle>, p: SolverParams, rng: Random, likelihood: Boolean): World? {
        val order = rec.deckOrder().toMutableList()
        val hidden = order.indices.filter { order[it] !in known }
        val shuffled = hidden.map { order[it] }.shuffled(rng)
        hidden.forEachIndexed { k, i -> order[i] = shuffled[k] }
        val h = rec.start(Deck.stacked(order, rec.variant.usesShortDeck))
        var w = 1.0
        for ((k, mv) in rec.moves.withIndex()) {
            val (seat, d) = mv
            if (h.toAct != seat) return null // the replay diverged (should not happen)
            if (likelihood && seat != hero && d !is Decision.Discard) {
                val ctx = h.context(seat)
                val bot = HouseBot(styles[seat], rng.nextLong() xor k.toLong(), p.botIterations, talk = false)
                if (category(ctx.legal.coerce(bot.decideNow(ctx).decision)) != category(d)) w *= p.mismatch
            }
            h.apply(seat, d)
        }
        return if (h.toAct == hero) World(order, w) else null
    }

    private fun playout(rec: HandRecord, world: World, hero: Int, move: Decision, styles: List<BotStyle>, p: SolverParams, seed: Long): Int {
        // the opponents' rollout policies for this world (same for every candidate move)
        val pick = Random(seed)
        val styles = styles.mapIndexed { i, st -> if (i == hero || pick.nextDouble() < p.ownStyle) st else POPULATION.random(pick) }
        val h = rec.start(Deck.stacked(world.order, rec.variant.usesShortDeck))
        for ((seat, d) in rec.moves) h.apply(seat, d)
        h.apply(hero, h.legalActions(hero).coerce(move))
        val bots = styles.indices.map { HouseBot(styles[it], seed * 31 + it, p.botIterations, talk = false) }
        var guard = 0
        while (!h.isFinished && guard++ < 400) {
            val seat = h.toAct ?: break
            val ctx = h.context(seat)
            h.apply(seat, ctx.legal.coerce(bots[seat].decideNow(ctx).decision))
        }
        return h.result?.finalChips?.get(hero) ?: rec.chips[hero]
    }

    fun solve(rec: HandRecord, h: Hand, hero: Int, styles: List<BotStyle>, persona: BotStyle?, p: SolverParams = SolverParams(), seed: Long = 0): SolverResult {
        val ctx = h.context(hero)
        val cands = candidates(ctx)
        val bb = ctx.bigBlind.toDouble()
        if (cands.size == 1) return SolverResult(cands[0], mapOf(cands[0] to 0.0), mapOf(cands[0] to 0.0), 0, 0.0)
        val known = (ctx.holeCards + ctx.board + ctx.seats.flatMap { it.cards.filterNotNull() }).toSet()
        // more candidates (draw-game discards) -> fewer worlds each
        val worlds = if (cands.size > 8) max(48, p.worlds / 3) else p.worlds
        val perWorld = (0 until worlds).toList().parallelStream().map { w ->
            val rng = Random(seed * 1_000_003 + w)
            val world = sampleWorld(rec, hero, known, styles, p, rng, likelihood = true) ?: return@map null
            val ws = rng.nextLong()
            world.weight to DoubleArray(cands.size) { c -> playout(rec, world, hero, cands[c], styles, p, ws).toDouble() }
        }.toList().filterNotNull()
        val wsum = perWorld.sumOf { it.first }
        val ess = wsum * wsum / perWorld.sumOf { it.first * it.first }
        val ev = DoubleArray(cands.size) { c -> perWorld.sumOf { it.first * it.second[c] } / wsum }
        // Standard error of each move's edge over the passive move (check, else fold, else the
        // first discard option), paired world by world (common random numbers).
        val passive = cands.indexOfFirst { it == Decision.Check }.takeIf { it >= 0 } ?: cands.indexOfFirst { it == Decision.Fold }.coerceAtLeast(0)
        val se = DoubleArray(cands.size) { c ->
            val m = ev[c] - ev[passive]
            val v = perWorld.sumOf { (w, xs) -> w * (xs[c] - xs[passive] - m).let { it * it } } / wsum
            kotlin.math.sqrt(v / ess)
        }
        val best = ev.max()
        val evBb = cands.indices.associate { cands[it] to (ev[it] - best) / bb }
        val seBb = cands.indices.associate { cands[it] to se[it] / bb }
        // Lower confidence bound: a move must beat the passive one by more than its noise.
        // Then the persona's style breaks near-ties (bonuses are fractions of a big blind).
        val choice = cands[cands.indices.maxBy { c ->
            val d = cands[c]
            val lcb = (ev[c] - p.confidence * se[c]) / bb
            if (persona == null) lcb else {
                val agg = if (category(d) == 2) 1.0 else 0.0
                val fold = if (d == Decision.Fold) 1.0 else 0.0
                lcb + p.styleBonusBb * ((persona.aggression - 0.5) * agg + (persona.tightness - 0.5) * fold + persona.bluffing * agg)
            }
        }]
        return SolverResult(choice, evBb, seBb, perWorld.size, ess)
    }
}
