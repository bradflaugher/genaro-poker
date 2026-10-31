// Genaro's bridge into PokerSI: compiled together with PokerSI's own pure-Kotlin engine and
// prompt code (see build.sh), so every prompt, grammar, blur and parse is byte-for-byte the
// app's. Three modes:
//
//   dump  --hands N --seed S --out decisions.jsonl   self-play tables, one record per decision
//   blur  < lines.jsonl > blurred.jsonl                run HandBlur on candidate table talk
//   play  --server URL --hands N --seed S --out f      one seat played by a llama.cpp server
//                                                      vs House Bots, exactly like LlmAgent
package genaro

import com.bradflaugher.pokersi.engine.*
import com.bradflaugher.pokersi.llm.*
import java.io.File
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import kotlin.coroutines.Continuation
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.coroutines.startCoroutine
import kotlin.random.Random

// ---------------------------------------------------------------------------- tiny JSON
fun q(s: String?): String {
    if (s == null) return "null"
    val b = StringBuilder("\"")
    for (c in s) when {
        c == '"' -> b.append("\\\"")
        c == '\\' -> b.append("\\\\")
        c == '\n' -> b.append("\\n")
        c == '\r' -> b.append("\\r")
        c == '\t' -> b.append("\\t")
        c < ' ' -> b.append("\\u%04x".format(c.code))
        else -> b.append(c)
    }
    return b.append('"').toString()
}
fun obj(vararg kv: Pair<String, Any?>): String = kv.joinToString(",", "{", "}") { (k, v) -> q(k) + ":" + js(v) }
fun js(v: Any?): String = when (v) {
    null -> "null"
    is String -> q(v)
    is Number, is Boolean -> v.toString()
    is RawJson -> v.s
    is Collection<*> -> v.joinToString(",", "[", "]") { js(it) }
    else -> q(v.toString())
}
class RawJson(val s: String)

/** Minimal JSON reader (objects, arrays, strings, numbers, bools, null). */
class JsonReader(private val s: String) {
    private var i = 0
    fun read(): Any? { ws(); return value() }
    private fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
    private fun value(): Any? {
        ws()
        return when (s[i]) {
            '{' -> { i++; val m = LinkedHashMap<String, Any?>(); ws(); if (s[i] == '}') { i++; return m }
                while (true) { ws(); val k = str(); ws(); i++; m[k] = value(); ws(); if (s[i++] == '}') break }; m }
            '[' -> { i++; val l = ArrayList<Any?>(); ws(); if (s[i] == ']') { i++; return l }
                while (true) { l.add(value()); ws(); if (s[i++] == ']') break }; l }
            '"' -> str()
            't' -> { i += 4; true }
            'f' -> { i += 5; false }
            'n' -> { i += 4; null }
            else -> { val st = i; while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++; s.substring(st, i).toDouble() }
        }
    }
    private fun str(): String {
        i++
        val b = StringBuilder()
        while (s[i] != '"') {
            if (s[i] == '\\') {
                i++
                when (s[i]) {
                    'n' -> b.append('\n'); 't' -> b.append('\t'); 'r' -> b.append('\r'); 'b' -> b.append('\b'); 'f' -> b.append('\u000c')
                    'u' -> { b.append(s.substring(i + 1, i + 5).toInt(16).toChar()); i += 4 }
                    else -> b.append(s[i])
                }
            } else b.append(s[i])
            i++
        }
        i++
        return b.toString()
    }
}

/** The engine's agents are `suspend` but never actually suspend: run them inline. */
fun <T> blocking(block: suspend () -> T): T {
    var out: Result<T>? = null
    block.startCoroutine(Continuation(EmptyCoroutineContext) { out = it })
    return out!!.getOrThrow()
}

// ---------------------------------------------------------------------------- shared helpers
fun styleOf(p: Persona) = BotStyle("persona", p.tightness.toDouble(), p.aggression.toDouble(), p.bluffiness * 0.35, 0.6)

fun personaFor(rng: Random): Persona =
    if (rng.nextDouble() < 0.2) Personas.handWritten.random(rng) else Personas.generate(rng.nextLong())

/** A decision as the app's grammar would write it: word + amount, or 1-based discard positions. */
fun asOutput(d: Decision, ctx: DecisionContext): Map<String, Any?> {
    if (d is Decision.Discard) return mapOf("discard" to d.indices.map { it + 1 }.sorted())
    val menu = PokerGrammar.actionMenu(ctx)
    val allin = menu.firstOrNull { it.word == "allin" }
    return when (d) {
        Decision.Fold -> mapOf("action" to "fold", "amount" to 0)
        Decision.Check -> mapOf("action" to "check", "amount" to 0)
        Decision.Call -> mapOf("action" to "call", "amount" to ctx.legal.toCall)
        Decision.AllIn -> if (allin != null) mapOf("action" to "allin", "amount" to allin.amount)
            else mapOf("action" to "raise", "amount" to (ctx.legal.maxRaiseTo ?: 0))
        is Decision.RaiseTo -> {
            val raise = menu.firstOrNull { it.word == "raise" }
            when {
                allin != null && (raise == null || d.total >= allin.amount) -> mapOf("action" to "allin", "amount" to allin.amount)
                raise?.range != null -> mapOf("action" to "raise", "amount" to d.total.coerceIn(raise.range!!.first, raise.range!!.last))
                raise != null -> mapOf("action" to "raise", "amount" to raise.amount)
                else -> mapOf("action" to "call", "amount" to ctx.legal.toCall)
            }
        }
        is Decision.Discard -> error("handled above")
    }
}

fun faceUpOf(h: Hand, seat: Int): List<Card> {
    val mine = h.snapshot(seat).seats[seat]
    return mine.cards.filterIndexed { k, _ -> mine.faceUp.getOrElse(k) { false } }.filterNotNull()
}

fun blurFor(ctx: DecisionContext, faceUp: List<Card>): (String) -> String = {
    HandBlur.blur(it, ctx.holeCards, faceUp, ctx.madeHand, false, ctx.board, ctx.variant.usesShortDeck, !ctx.variant.mustUseTwoHoleCards)
}

fun equityOf(ctx: DecisionContext, iterations: Int, rng: Random): Double {
    val opps = ctx.seats.filter { it.seat != ctx.seat && !it.folded }
    val ups = opps.map { s -> s.cards.filterNotNull() }
    return Equity.estimate(ctx.holeCards, ctx.board, opps.size, ctx.variant, iterations, rng, ups)
}

fun args(a: Array<String>): Map<String, String> =
    a.toList().windowed(2, 2, partialWindows = false).filter { it[0].startsWith("--") }.associate { it[0].drop(2) to it[1] }

fun randomTable(rng: Random): Pair<GameVariant, List<Persona>> {
    val variant = GameVariant.entries.random(rng)
    val n = rng.nextInt(2, 7)
    val ps = mutableListOf<Persona>()
    while (ps.size < n) { val p = personaFor(rng); if (ps.none { it.name == p.name }) ps += p }
    return variant to ps
}

fun stakesFor(variant: GameVariant, rng: Random): Pair<Stakes, Int> {
    val bb = listOf(10, 20, 40, 50, 100, 200).random(rng)
    val ante = if (variant == GameVariant.SEVEN_CARD_STUD) maxOf(1, bb / 10) else 0
    return Stakes.forVariant(variant, bb / 2, bb, ante) to bb
}

// ---------------------------------------------------------------------------- dump
fun dump(a: Map<String, String>) {
    val hands = a["hands"]!!.toInt()
    val rng = Random(a["seed"]!!.toLong())
    val out = File(a["out"]!!).bufferedWriter()
    val iters = (a["iters"] ?: "2000").toInt()
    // Optional: for these record ids, also ask K independent teachers and keep the majority
    // decision (a denoised label). Votes use their own RNGs, so the records are unchanged.
    val voteIds = a["vote-ids"]?.let { File(it).readLines().toHashSet() } ?: emptySet()
    val votes = (a["votes"] ?: "9").toInt()
    var records = 0
    var handIx = 0
    while (handIx < hands) {
        val (variant, personas) = randomTable(rng)
        val talk = TableTalk()
        val lastKind = HashMap<String, String?>()
        // the teacher: House Bot logic with each persona's style and a sharp equity estimate
        val teachers = personas.map { HouseBot(styleOf(it), rng.nextLong(), iters, talk = true, tableTalk = talk) }
        val (stakes, bb) = stakesFor(variant, rng)
        var chips = personas.map { bb * listOf(25, 50, 100, 100, 150, 250).random(rng) }
        val handsHere = rng.nextInt(3, 9)
        for (k in 0 until handsHere) {
            if (handIx >= hands || chips.count { it > 0 } < 2) break
            handIx++
            val h = Hand(variant, personas.indices.map { SeatSetup(personas[it].name, chips[it]) },
                button = k % personas.size, stakes = stakes, seed = rng.nextLong(), handNumber = k + 1)
            var guard = 0
            while (!h.isFinished && guard++ < 500) {
                val seat = h.toAct ?: break
                val ctx = h.context(seat)
                val p = personas[seat]
                val plan = TalkPlan.make(ctx, p, talk, rng, lastKind[p.name])
                lastKind[p.name] = plan.kind
                val msgs = PokerPrompt.build(ctx, p, plan)
                val grammar = PokerGrammar.forContext(ctx, plan.speak)
                val eq = if (ctx.legal.discard == null) equityOf(ctx, iters, rng) else null
                val teacher = teachers[seat].decideNow(ctx)
                val decision = ctx.legal.coerce(teacher.decision)
                val me = ctx.seats.first { it.seat == seat }
                val id = "s${a["seed"]}-$records"
                val majority = if (id in voteIds) majorityOf(ctx, p, id, votes) else null
                out.write(obj(
                    "id" to id, "variant" to variant.name, "street" to ctx.street,
                    "hand" to ctx.handNumber, "seats" to personas.size, "seat" to seat,
                    "persona" to RawJson(obj("name" to p.name, "voice" to p.voice, "bio" to p.bio, "catchphrase" to p.catchphrase,
                        "tightness" to p.tightness, "aggression" to p.aggression, "bluffiness" to p.bluffiness, "temperature" to p.temperature)),
                    "speak" to plan.speak, "hint" to plan.hint, "kind" to plan.kind, "mine" to plan.mine,
                    "others" to plan.others.map { RawJson(obj("speaker" to it.speaker, "text" to it.text)) },
                    "system" to msgs[0].content, "user" to msgs[1].content, "grammar" to grammar,
                    "discard" to (ctx.legal.discard != null),
                    "discard_min" to ctx.legal.discard?.min, "discard_max" to ctx.legal.discard?.max,
                    "can_check" to ctx.legal.canCheck, "to_call" to ctx.legal.toCall,
                    "min_raise" to ctx.legal.minRaiseTo, "max_raise" to ctx.legal.maxRaiseTo,
                    "menu" to PokerGrammar.actionMenu(ctx).map { RawJson(obj("word" to it.word, "amount" to it.amount, "lo" to it.range?.first, "hi" to it.range?.last)) },
                    "pot" to ctx.pot, "bb" to ctx.bigBlind, "chips" to me.chips, "bet" to me.betThisRound,
                    "opponents" to ctx.seats.count { it.seat != seat && !it.folded },
                    "equity" to eq, "made_hand" to ctx.madeHand,
                    "hole" to ctx.holeCards.map { it.toString() }, "board" to ctx.board.map { it.toString() },
                    "face_up" to faceUpOf(h, seat).map { it.toString() },
                    "short_deck" to variant.usesShortDeck, "board_plays" to !variant.mustUseTwoHoleCards,
                    "teacher" to RawJson(obj(*asOutput(decision, ctx).map { it.key to it.value }.toTypedArray())),
                    "bot_say" to teacher.say,
                    "majority" to majority?.let { RawJson(obj(*it.map { e -> e.key to e.value }.toTypedArray())) },
                ))
                out.newLine()
                records++
                if (teacher.say != null && plan.speak) talk.record(p.name, teacher.say!!)
                h.apply(seat, decision, teacher.say)
            }
            chips = personas.indices.map { h.result?.finalChips?.getOrNull(it) ?: chips[it] }
            if (chips.count { it > 0 } < 2) chips = chips.map { if (it == 0) bb * 100 else it }
        }
    }
    out.close()
    System.err.println("dumped $records decisions from $handIx hands")
}

/** The most common decision among [k] independent teachers with this persona's style. */
fun majorityOf(ctx: DecisionContext, p: Persona, id: String, k: Int): Map<String, Any?> {
    val outs = (0 until k).map { j ->
        val bot = HouseBot(styleOf(p), (id.hashCode().toLong() shl 8) + j, 1000, talk = false)
        asOutput(ctx.legal.coerce(bot.decideNow(ctx).decision), ctx)
    }
    val key = { o: Map<String, Any?> -> (o["action"] ?: o["discard"].toString()).toString() }
    val top = outs.groupBy(key).maxBy { it.value.size }
    val pick = top.value
    if (pick.first().containsKey("discard")) return pick.first() + mapOf("votes" to pick.size)
    val amounts = pick.map { (it["amount"] as Int) }.sorted()
    return mapOf("action" to pick.first()["action"], "amount" to amounts[amounts.size / 2], "votes" to pick.size)
}

// ---------------------------------------------------------------------------- blur
fun blur() {
    val out = System.out.bufferedWriter()
    generateSequence(::readLine).forEach { line ->
        val r = JsonReader(line).read() as Map<*, *>
        fun cards(k: String) = (r[k] as List<*>).map { Card.parse(it as String) }
        val text = r["text"] as String
        val b = HandBlur.blur(text, cards("hole"), cards("face_up"), r["made_hand"] as String?, false, cards("board"),
            r["short_deck"] as Boolean, r["board_plays"] as Boolean)
        out.write(obj("id" to r["id"], "text" to text, "blurred" to b, "leaks" to HandBlur.blurred(b)))
        out.newLine()
    }
    out.flush()
}

// ---------------------------------------------------------------------------- play
class ServerAgent(
    private val url: String, private val persona: Persona, private val talk: TableTalk, private val rng: Random,
    private val stats: MutableMap<String, Double>, private val log: java.io.Writer?, private val fallback: HouseBot,
) {
    private val http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build()
    private var lastKind: String? = null

    fun decide(h: Hand, ctx: DecisionContext): AgentReply {
        val plan = TalkPlan.make(ctx, persona, talk, rng, lastKind)
        lastKind = plan.kind
        val msgs = PokerPrompt.build(ctx, persona, plan)
        val grammar = PokerGrammar.forContext(ctx, plan.speak)
        val body = obj(
            "messages" to msgs.map { RawJson(obj("role" to it.role, "content" to it.content)) },
            "grammar" to grammar, "temperature" to persona.temperature, "top_k" to 40, "top_p" to 0.95,
            "max_tokens" to 96, "seed" to rng.nextInt(Int.MAX_VALUE),
            "chat_template_kwargs" to RawJson(obj("enable_thinking" to false)),
        )
        val t0 = System.nanoTime()
        stats.merge("decisions", 1.0, Double::plus)
        val raw = try {
            val req = HttpRequest.newBuilder(URI.create("$url/v1/chat/completions")).timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build()
            val resp = JsonReader(http.send(req, HttpResponse.BodyHandlers.ofString()).body()).read() as Map<*, *>
            val usage = resp["usage"] as Map<*, *>
            stats.merge("prompt_tokens", (usage["prompt_tokens"] as Double), Double::plus)
            stats.merge("completion_tokens", (usage["completion_tokens"] as Double), Double::plus)
            (((resp["choices"] as List<*>)[0] as Map<*, *>)["message"] as Map<*, *>)["content"] as String
        } catch (e: Exception) { stats.merge("errors", 1.0, Double::plus); "" }
        stats.merge("ms", (System.nanoTime() - t0) / 1e6, Double::plus)
        val reply = LlmReply.parse(raw, ctx.legal, ctx.holeCards.size)
        val decision = reply.decision ?: run { stats.merge("fallbacks", 1.0, Double::plus); fallback.decideNow(ctx).decision }
        var say = reply.say?.trim()?.takeIf { it.isNotEmpty() && plan.speak }
        if (say != null) {
            stats.merge("lines", 1.0, Double::plus)
            stats.merge("words", say.split(Regex("\\s+")).size.toDouble(), Double::plus)
            if (say.split(Regex("\\s+")).size > 12) stats.merge("long_lines", 1.0, Double::plus)
            if (HandBlur.blurred(blurFor(ctx, faceUpOf(h, ctx.seat))(say))) stats.merge("leaks", 1.0, Double::plus)
            if (talk.isRepeat(say, persona.name)) { stats.merge("repeats", 1.0, Double::plus); say = null }
        }
        if (plan.speak && say == null && reply.say == null) stats.merge("silent_on_talk_turn", 1.0, Double::plus)
        log?.apply { write(obj("hand" to ctx.handNumber, "street" to ctx.street, "speak" to plan.speak, "raw" to raw,
            "decision" to decision.toString(), "say" to say)); write("\n") }
        return AgentReply(decision, say)
    }
}

fun play(a: Map<String, String>) {
    val url = a["server"]!!
    val hands = a["hands"]!!.toInt()
    val seed = a["seed"]!!.toLong()
    // seat 0 = the model (or a baseline: "housebot" / "advice"); seats 1-3 = House Bots
    val mode = a["mode"] ?: "server"
    val stats = LinkedHashMap<String, Double>()
    val log = a["log"]?.let { File(it).bufferedWriter() }
    var net = 0.0
    var bbSum = 0.0
    var bbSq = 0.0
    for (k in 0 until hands) {
        // identical deals, personas and opponents for every model at the same (seed, k)
        val rng = Random(seed * 1_000_003 + k)
        val variant = GameVariant.entries[k % GameVariant.entries.size]
        val personas = List(4) { personaFor(rng) }.mapIndexed { i, p -> if (i == 0) p else p.copy(name = p.name + " " + i) }
        val talk = TableTalk()
        val bots = personas.map { HouseBot(styleOf(it), rng.nextLong(), 250, talk = true, tableTalk = talk) }
        val (stakes, bb) = stakesFor(variant, rng)
        val start = bb * 100
        val h = Hand(variant, personas.map { SeatSetup(it.name, start) }, button = k % 4, stakes = stakes, seed = rng.nextLong(), handNumber = k + 1)
        val agentRng = Random(seed + k)
        val model = ServerAgent(url, personas[0], talk, agentRng, stats, log, HouseBot(styleOf(personas[0]), 1, 250, talk = false))
        var guard = 0
        while (!h.isFinished && guard++ < 500) {
            val seat = h.toAct ?: break
            val ctx = h.context(seat)
            val reply = when {
                seat != 0 -> bots[seat].decideNow(ctx)
                mode == "server" -> model.decide(h, ctx)
                mode == "housebot" -> bots[0].decideNow(ctx)
                else -> AgentReply(adviceDecision(ctx, personas[0]))
            }
            val d = ctx.legal.coerce(reply.decision)
            reply.say?.let { talk.record(personas[seat].name, it) }
            h.apply(seat, d, reply.say)
        }
        val delta = (h.result?.finalChips?.get(0) ?: start) - start
        net += delta
        bbSum += delta.toDouble() / bb
        bbSq += (delta.toDouble() / bb) * (delta.toDouble() / bb)
        if ((k + 1) % 50 == 0) System.err.println("  ${k + 1}/$hands hands, ${"%.1f".format(100 * bbSum / (k + 1))} bb/100")
    }
    log?.close()
    stats["hands"] = hands.toDouble()
    stats["bb_per_100"] = 100 * bbSum / hands
    stats["bb_sum"] = bbSum
    stats["bb_sq"] = bbSq
    File(a["out"]!!).writeText(obj(*stats.map { it.key to it.value }.toTypedArray()) + "\n")
    System.err.println(stats)
}

/** The baseline a model gets for free: always do what the prompt's Advice line says. */
fun adviceDecision(ctx: DecisionContext, p: Persona): Decision {
    if (ctx.legal.discard != null) return Decision.Discard(emptySet())
    val user = PokerPrompt.build(ctx, p, null)[1].content
    val word = Regex("Advice: \"(\\w+)\"").find(user)?.groupValues?.get(1) ?: "check"
    return when (word) {
        "fold" -> Decision.Fold; "check" -> Decision.Check; "call" -> Decision.Call; "allin" -> Decision.AllIn
        else -> Decision.RaiseTo(ctx.legal.minRaiseTo ?: 0)
    }
}

fun main(argv: Array<String>) {
    when (argv.firstOrNull()) {
        "dump" -> dump(args(argv.drop(1).toTypedArray()))
        "blur" -> blur()
        "play" -> play(args(argv.drop(1).toTypedArray()))
        else -> System.err.println("usage: bridge dump|blur|play ...")
    }
}
