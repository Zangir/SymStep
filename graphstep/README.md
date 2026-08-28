# GraphStep: one general reasoner, zero LLM calls, every answer certified

GraphStep is a **general reasoning system**: one algorithm that reads any
input — a logic puzzle, a coding task, a story, a theory, or a random
paragraph of prose — turns what it understood into formal structure, reasons
over it symbolically, and returns either a **certified answer** or a
**precise refusal** naming what it could not read. No LLM is involved
anywhere; on every benchmark below the LLM call count is exactly zero.

Two commitments define the project:

**Generalization is the goal, not a style preference.** There is no
per-task code anywhere: no benchmark scripts, no dataset manifests, no
task-typed branches. A benchmark is only *where samples come from*. What a
new task needs is new *knowledge* (rows in one store) or a new *general
mechanism* (a capability any input can use) — never new control flow. This
is enforced mechanically: `tests/test_generality.py` fails if a benchmark
name or a task-typed parameter appears in any core module.

**Honesty is structural, not aspirational.** Answers carry certificates
(a uniqueness proof, a passing test suite, a proof chain). Anything the
system could not read is reported, never guessed over. When the evidence
is ambiguous, it abstains. In a full audit of ~23,400 scored samples, every
one of the 8 answers that disagreed with the official answer key traced to
documented *label noise in the datasets* — with training-data evidence —
not to the engine.

## The one algorithm (six steps, every sample identical)

```
sample (any source, any shape)
  1 RECOGNIZE  find shapes in the raw input: test-code lines -> typed I/O
               examples; value grids -> puzzle groups; numbered lines ->
               premises; "(A) ..." lines -> options; "?" -> the question;
               answer-shaped content (a program, a filled table) -> QUARANTINED
  2 PARSE      grammar: who did what to whom, negation, conditions, tense
  3 GROUND     every word/statement -> rows of the ONE knowledge store
               (verbs, selectors, clue shapes, event verbs, assembly blocks);
               WordNet widens on a miss, with provenance recorded
  4 EMIT       everything grounded becomes ONE formal problem: puzzle groups
               license position variables; statements license constraints;
               I/O examples + a grounded frame license PROGRAM-SLOT variables
               (program synthesis IS constraint solving, on the same engine);
               narratives fold into time-stamped fluent edges
  5 SOLVE      propagation + search + certification (engine/core.py), or
               forward-chaining closure with proof chains, or fluent queries
  6 ANSWER     certified result in the text's own words — or a named refusal
               (UNGROUNDED / AMBIGUOUS / UNSAT / UNDETERMINED / EXCLUDED)
```

The only dispatch anywhere is on **evidence shape** (what the sample
contains) and **artifact type** (what kind of formal object was built) —
never on where the sample came from.

## Repository layout

```
graphstep/
  unified.py        the one algorithm (the six steps above)
  run.py            the one entry point: any HF dataset or JSON file
  engine/           the reasoning substrate (fully task-agnostic)
    core.py         finite-domain engine: propagation, search, uniqueness
                    certificates, explanation ledger, UNSAT cores
    constraints.py  typed constraint library incl. reified combinators
    ir.py           the JSON logic IR the compilers target
    sandbox.py      the execution oracle for the program algebra
  reading/          the reading layer (all domain knowledge lives here as DATA)
    kb.py           THE knowledge store: one relation of rows — verbs,
                    selectors, role types, clue shapes, event verbs, query
                    shapes, assembly blocks; WordNet-widened, provenance-stamped
    semgraph.py     total-capture clause reader: every clause of any text ->
                    typed roles, tense, polarity, ATTRIBUTION ("the CEO said X"
                    is a fact about the CEO, never about X); two trust tiers
    narrative.py    temporal fluent layer: events fold into NEVER-DELETED
                    interval-stamped edges (a move closes the old location
                    edge); value queries walk current state or history
    worldmodel.py   open-prose facts + quantified rules + closure with proofs
    universal.py    ontology induction from raw text (no given inventory)
    compile_text.py lexical mention grounding (surface-variant normalization)
    syntax_tier.py  clause skeletons: if/then/unless, coordination, negation
    ordering.py     linear-order reading over antonym scales (left/right,
                    cheap/expensive, new/old -> one abstract axis)
    retrieval.py    knowledge gaps -> WordNet/Wikidata, admitted as symbols
    kbstore.py      persistent SQLite cache of retrieved knowledge
    induction.py    rule mining over fact stores, verified before use
  tests/            CI: generality (name/branch bans), semgraph honesty,
                    parity vs gold, plus unit tests
  legacy/           superseded per-task scripts, kept ONLY as parity references
  results/          benchmark result files
  data/             runtime caches (not committed)
```

## Honesty mechanisms, concretely

- **Certificates.** A puzzle answer means "unique solution, every constraint
  satisfied, proven by exhaustive search". A program means "assembled from
  spec-licensed parts AND passes every test". A derived fact carries its
  proof chain.
- **Two tiers of trust** (`semgraph.py`). EVERY clause of any text enters
  the graph ("captured" — searchable, reportable). Only clauses a knowledge
  row actually understands are promoted to "grounded" and may appear in
  proofs. Conclusions never cite captured-only material.
- **Attribution.** Reporting verbs (say, claim, expect, deny ...) wrap their
  content: the system will conclude "the spokesman claims the water is safe"
  and will not conclude "the water is safe".
- **Nothing is deleted.** Time-varying facts get interval stamps; a new
  value closes the old edge rather than erasing it, so history stays
  queryable and two simultaneously-open exclusive edges are a detected
  contradiction, not an update.
- **Calibration, never hand-guessing.** Template meanings and world
  knowledge (spatial phrase vectors, "thirsty people head to the kitchen")
  are MINED from labeled training data with provenance
  (`calibrated:<source>`); test data is never touched.
- **Quarantine.** Anything answer-shaped in the input (reference programs,
  solution tables) is recognized and locked away unused.

## Results (full suite, 2026-08, zero LLM calls everywhere)

| benchmark | result | wrong |
|---|---|---|
| LGP-20 + SP-6 grid/scheduling puzzles (vs gold) | 26/26 | 0 |
| ZebraLogicBench, all sizes (1000) | 959 certified, 41 excluded (encoding) | 0 |
| BBH logical_deduction 3/5/7 objects (750) | 750/750 | 0 |
| BBH web_of_lies (250) | 250/250 | 0 |
| ProofWriter depths 0-5 + ext + birds-electricity (800) | 800/800; paraphrased split abstains | 0 |
| StepGame spatial (500) | 387 correct, 112 abstain | 1* |
| Story QA, 20 tasks (20,000) | 19,238 correct + 755 abstain honest; 19,654/20,000 closed-world | 7* |
| MBPP, internal reasoning only | **33 certified**, 7 proven-UNSAT | 0 |
| MBPP + worked-example corpora (~438k pairs) | additional solves graded verified / adapted / mined, behind the license stack below; spot-audited case by case | 0 found |
| AR-LSAT (230) | honest refusals — frame reading not built yet | 0 |

\* All 8 disagreements are documented dataset-label noise: the StepGame gold
is geometrically impossible given its own story (hand-checkable); the story-QA
generator is provably inconsistent on multi-give questions (training gold
follows "last give" 11 times, "first give" 2 times) and on color induction
(no policy explains training gold better than the one used, 99.4%).

## Reasoning over any text (not just benchmarks)

Give `unified.solve` a plain paragraph and no question, and it returns what
logically FOLLOWS, each conclusion with its proof:

> "Anne is kind. If someone is kind then they help the team. The manager
> claimed Anne is lazy."
> → **proves** `anne helps the team` (rule + "Anne is kind")
> → **attributes** `manager claims: "anne is lazy"` (never asserted)

Narratives work the same way through the fluent layer:

> "Elena picked up the violin. Elena went to the vault. Elena dropped the
> violin. She went back to the rehearsal."
> → Where is the violin? **vault**. Where is Elena? **rehearsal**.

Ordinary news prose is *captured* (every clause, with roles and tense) and
reported honestly as not-yet-proof-grade — the reading frontier is measured
per text as `grounded / captured`, and it grows row by row, precision-first.

## Running it

```
pip install -r requirements.txt          # datasets, spacy, z3, nltk
python3 -m spacy download en_core_web_sm

# any dataset, no manifest — the sample's shapes decide everything:
python3 -m graphstep.run --hf <org/dataset> [--config X] [--split test]
    [--gold <answer-field>] [--question <query-field>] [--drop <gold fields>]
    [--calibrate module:function[:source@split]] [--limit N] [--per FIELD=N]
    [--loop]        # the solver loop: decompose / derive / retrieve / learn

# arm worked-example corpora (any dataset of (description, code) pairs):
#   --calibrate "graphstep.reading.sources:register_procedure_corpus:\
#                <org/dataset>[#config]@<split>[+split]:<textfield>:<codefield>"

# one sample from JSON / stdin:
python3 -m graphstep.unified sample.json

# the CI gates:
python3 graphstep/tests/test_generality.py      # no per-task code, ever
python3 graphstep/tests/test_semgraph.py        # capture/attribution honesty
python3 graphstep/tests/test_unified_parity.py  # puzzle results vs gold
python3 graphstep/tests/test_compose.py         # compiler algebra-neutrality
python3 graphstep/tests/test_sources.py         # retrieval kinds + tribunal
```

## Gap-driven retrieval (`reading/sources.py`)

**The knowledge-kind ontology** (`KINDS`): a CLOSED set — meaning,
callable, formula, fact, procedure, example, policy — where every kind
declares its representation, its admission oracle, the GRADE its use
confers (certified / verified / likely / calibrated), and whether it may
persist. No oracle, no kind. Procedures are task-shaped: a worked example
from any registered corpus is matched to the spec, its entry point aliased,
and it is employed ONLY if it passes the task's own oracle — graded
"verified", never "certified", and never stored. The CI proves the gating
with a trap: a wrong candidate with a better lexical match must lose to a
correct one with a worse match.

A complete atom set is impossible; knowledge is PULLED BY THE GAP instead.
A refusal names its missing word; that name becomes a Gap; SOURCE ADAPTERS
answer it with candidate rows in the one Row schema, provenance-stamped;
the ADMISSION TRIBUNAL rejects anything a higher-trust row already defines
(hand = calibrated > derived = introspected > WordNet/Wikidata > web);
EMPLOYMENT stays oracle-gated — a retrieved meaning is used only if the
composition it enters survives the tests or proofs. Nothing is absorbed
as text.

Two adapters ship: **introspection** (the Python runtime is interrogated
for real callables and their true signatures — the entire standard library
becomes an on-demand atom space: "find the gcd of two numbers" retrieves
math.gcd, passes the oracle, and is certified with provenance
`introspected:math.gcd`) and the existing **WordNet/Wikidata** route for
the claim algebra ("Rex is a wolf" + "Rex is a canine?" -> True, graded
`likely`, proof quoting `[retrieved:wordnet] every wolf is a canine`).
Background-knowledge retrieval is OPT-IN per run (synthetic theory
benchmarks must not be answered from real-world knowledge); retrieved-tier
answers are always GRADED, never silently mixed with certified ones.
`tests/test_sources.py` is CI: real callables found, hand rows win,
oracle-gated employment, honest refusal on unknown words, grading+opt-in.

**Focused requests, not similarity.** A worked-example query is the
CONJUNCTION of the task's discriminative words (generic words — type
nouns, operation verbs, function words — carry no identity): a candidate
containing some form of EVERY distinctive word is fetched no matter how
the rest of its wording differs; lexical overlap only ranks within the
fetched set. Words match by VARIANT SETS ({co-prime, coprime},
{checks, check}) — surface and normalized forms are all searched, so
nothing is ever lost to spelling.

**The imposter license stack** (each entry added after a real caught
imposter): discriminative-word coverage (a candidate lacking every form
of one of the task's distinctive terms cannot mean the task — a
prime-checker passing a Woodall task's asserts); recognized-conflict
rejection (a shared word grounding to a DIFFERENT meaning, min vs max);
negation-marker agreement (with "...or not" excluded as an idiom);
independent-argument coverage (a variant dropping an independent varying
input is refused, with length-affine arguments exempt as dependent);
and mining restricted to discriminative key words. Zero wrong answers is
the maintained invariant; grades (certified / verified / adapted / mined)
keep every claim auditable.

**The code reader** (`reading/codereader.py`): retrieved CODE passes
through our own reader, as retrieved prose must — AST decomposition into
typed rows (single-return expressions -> predicates/functions/operators;
conditional-accumulator loops -> the predicate inside), with a purity
whitelist and the capture-tier rule: unrecognized shapes yield nothing.
Mined meanings compose through the same compiler under the same licenses.

Corpora scale via an inverted variant-word index (lookup independent of
corpus size; ~438k pairs registered across four corpora). Measured lesson:
**source fit beats source size** — 18k instruction-phrased examples
contributed as many solves as the benchmark's own training split; 400k
API-doc functions contributed three. Arming corpora does not touch any
other reasoning path (verified empirically: puzzles, theories, stories,
and truth-teller chains are bit-identical with sources armed).

## The solver loop (`agenda.py`)

Above the six steps sits a general while-loop that works a task part by
part — the knowledge escalation ladder run per subgoal:

```
while open goals remain:
    GROUND      known rows / the one-shot pipeline answer the goal
    DECOMPOSE   split it by structure into subgoals
    RETRIEVE    ask an external source, admit typed rows (provenance)
    DERIVE      spec-guided composition of ATOMS for this subgoal only;
                every candidate faces the oracle
    VERIFY      tests / uniqueness proofs / closure judge the artifact
    REPAIR      a failure names the guilty subgoal; reopen exactly it
    LEARN       a verified composition becomes a new row — tomorrow's
                GROUND is today's DERIVE (session scope; persistence
                must be earned: verified + reusable + provenance-stamped)
    REFUSE      nothing applies -> a named gap, honestly closed
```

When knowledge suffices, the loop is one iteration (all benchmark parity
is preserved by construction). When it doesn't, the loop reasons down to
atoms — predicate atoms (empty, odd, prime, true, vowel, leap, ...),
reduction atoms (min, max, sum, average, ...), transform atoms (filter,
count, sort, reverse), binary relations (equal, greater) — composed by
schema per operation family and judged candidate-by-candidate in the
sandbox. Knowledge is scoped: bindings and story states die with the
task; only verified reusable compositions become rows.

Two honesty guards keep derivation from gaming weak test suites:
**head-licensing** (an atom may only fill the frame's head position —
"find the SMALLEST number" licenses min; "find the frequency of the
smallest value" does not) and **argument coverage** (a candidate must use
every INDEPENDENT varying argument — an argument derivable from another in
every example, like the classic redundant length argument n == len(arr),
is dependent and exempt). Both were added
after they caught real test-overfits ("co-prime" answered by primality of
one argument; lexicographic max passing a longest-sublist task) — and a
learned row carries its license with it, so reuse obeys the same gates.

Parse-first reading feeds all of this: when the parser mis-attaches the
action verb, structure is recovered lexicon-guided (first verb-like or
operation-grounded token after the imperative wrapper) — spec frames read
on MBPP went from 139/500 to 387/500, turning silent structural failures
into precisely named vocabulary gaps.

**The composition compiler (`reading/compose.py`)**: grammatical structure
determines how grounded meanings combine — "the sum OF the digits OF the
number" IS `sum(digits(n))`, because "of" means apply in every domain.
Composition devices (of-application, between-pairs, comparative
parameters, coordination, conditionals) are DEVICE rows in the one store;
domain enters through exactly one plug, the leaf grounder each algebra
supplies (proven algebra-neutral in `tests/test_compose.py`: the identical
compiler builds `sum(digits(arg0))` for code and `owner_of(DOG)` for a toy
relational algebra, and refuses unknown heads). A compiled tree is
licensed BY CONSTRUCTION — every atom sits exactly where the parse put its
word — which generalizes head-licensing to arbitrary depth.

Type COERCIONS extend composition beyond the sentence's surface: (BINOP,
SEQ) folds a binary atom across a sequence ("the gcd of the array
elements" -> `reduce(gcd, xs)` — open-class, proven in CI with a
runtime-plugged operator), and (APPLY, SEQ) maps an element-level chain
("the maximum sum of the sublists" -> `max([sum(e) for e in xs])`). The
KEY device ("sort BY / ACCORDING TO k") parameterizes by a compiled key
function. COLLECTION-RELATIVE predicates ("the sum of REPEATED elements":
a property of an element relative to its collection) ride modifiers into
the mapped comprehension's condition. All are rows; none knows a task.

Retrieved procedures can also be ADAPTED, not just replayed: when the
parsed diff between task and example is recognized, an adaptation operator
bridges it under the same oracle gate — the first shipped operator is
arity-fold ("merge THREE dictionaries" from a retrieved "merge two" ->
reduce(merge2, args)), graded "verified (adapted)". Corpus ingestion
extracts fenced code from prose-wrapped payloads (knowledge must be code,
not text about code).

## One task, end to end (how it all fits)

Task: *"check whether the given number is co-prime or not"* + three asserts.

1. The asserts reveal the signature: `is_coprime`, two integers, boolean.
2. The parse reads the frame (CHECK family) and the content words.
3. Internal derivation tries `is_prime` on one argument — the coverage
   guard refuses it (a two-input task answered from one input).
4. Word-gap retrieval (runtime, mining) finds no usable meaning.
5. The focused fetch queries the corpora with the variant group
   {co-prime, coprime} — the ONE matching example among 13k is fetched
   despite sharing almost no other wording with the task.
6. The license stack clears it (all distinctive words covered, no meaning
   conflicts, "...or not" recognized as an idiom, all independent
   arguments forwarded).
7. The sandbox runs it against the asserts: an Euclid gcd loop, `== 1`.
8. Answer: SOLVED, graded `verified (example-sourced, oracle-passed)` —
   not certified, because the program came from an example, and the
   grade says so. Nothing task-shaped enters the permanent store.

## Extending the system

Everything grows by adding **rows** or **general mechanisms**, never
branches:

1. New phrasing for a known meaning → one lexicon row (or free, via WordNet).
2. New operation/selector for code synthesis → one assembly-block row.
3. New sentence shape (an event verb family, a clue archetype, a question
   shape) → one row with an emitter.
4. New capability (a solver, an algebra, a temporal mechanism) → a general
   module any input can trigger by shape.

The growth metric that keeps the project honest: **rows added per newly
solved sample must fall over time**. If solving benchmark N+1 costs as much
new knowledge as benchmark 1 did, the generality claim is failing and the
metric will show it.

## Roadmap

- **AR-LSAT reading**: prose inventories ("the eight students—George, ..."),
  selection ("exactly six will report"), nested unless-conditionals, and
  schedule-shaped options — all as rows over the existing machinery.
- **Code synthesis growth** (MBPP → BigCodeBench → SWE-bench Verified):
  the example-kind gap (request a discriminating test when a weak oracle
  can't separate candidates — the principled cure for 3-assert tasks),
  typed queries (head + roles + argument types), loop/state composition
  devices, more instruction-style corpora, and live web sources through
  the same reader-and-license stack; the protocol is fixed — measure,
  explain every failure, the maintainer picks the fix, re-run, advance.
- **Reading enrichment**: promote captured clauses to proof grade in place
  (grounding bridge), causal/discourse rows ("because", "so"), WordNet
  is-a edges on every graph node, and induction over the accumulated graph.
