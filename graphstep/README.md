# GraphStep: one general reasoner, zero LLM calls, every answer checked

GraphStep is a reasoning system built around one rule: **never guess.**
It reads a task in plain English — a logic puzzle, a coding problem, a
story, a theory, a request for a summary — turns what it understood into
a formal structure, solves that structure symbolically, and returns either
a **checked answer** or a **refusal that names exactly what it could not
understand**. No language model is involved anywhere: on every benchmark
below, the LLM call count is zero.

Two commitments define the project:

**One engine, no per-task code.** There are no benchmark scripts, no
dataset switches, no "if this is a math task" branches. A benchmark is
only *where samples come from*. The engine looks at what a sample
*contains* (a value grid, input/output examples, prose, a request to
produce something) and routes it to the matching kind of answer. A CI
test fails the build if a benchmark name or a task-typed parameter ever
appears in a core module.

**Every answer says how much to trust it.** Answers carry certificates
(a uniqueness proof, a passing test suite, a proof chain) and grades that
name where the knowledge came from (`certified`, `verified`,
`calibrated`, `likely`). When the system finds a program that merely
*passes the tests by coincidence* but has no connection to the task's
words, it reports it as a **conjecture** — never as the answer. When it
cannot read something, it says which word or sentence stopped it.

## The one algorithm

Every sample, from any source, goes through the same six steps:

```
1 RECOGNIZE  what does the input contain? value grids -> a puzzle;
             input/output examples -> a program to build; prose ->
             claims or a story; "write a summary of X" -> a digest.
             Anything answer-shaped in the input is quarantined unused.
2 PARSE      read each sentence: who did what to whom, negation,
             conditions, tense.
3 GROUND     map every word and sentence to rows of ONE knowledge store.
             A word with no row becomes a NAMED GAP (see "questions").
4 EMIT       everything grounded becomes one formal problem: puzzle
             sentences become constraints; a spec plus examples becomes
             program slots; narrative sentences become time-stamped facts.
5 SOLVE      the matching solver: constraint propagation + search with a
             uniqueness proof, the test sandbox, forward-chaining with
             proof chains, or timeline queries.
6 ANSWER     a checked result in the text's own words — or a refusal
             naming what was missing (UNGROUNDED / AMBIGUOUS / UNSAT /
             UNDETERMINED / EXCLUDED).
```

Above these six steps sits a solver loop (`agenda.py`) that works a task
part by part: if one-shot solving fails, it decomposes the task, asks
questions about the parts it could not ground, requests knowledge from
sources, tries again, and finally refuses honestly. When existing
knowledge suffices, the loop is a single iteration.

## The question layer (`reading/questions.py`)

When the ordinary paths fail on a program-shaped task, the engine starts
asking typed questions about the *parts* of the task — the same way its
puzzle solver asks questions about variables:

- **DENOTE(phrase, type)** — what does this phrase mean, as the type the
  context demands? ("count ... substrings" demands a collection whose
  size is the answer.)
- **USAGE(word)** — show me this word used anywhere: registered
  documents, worked-example corpora. If many usages share one code shape
  (`word[0:2]`, `word[2:5]`, `text[s:e]`), that shared shape *is* the
  word's meaning and the varying numbers are its parameters.
- **BRIDGE(have, want)** — what converts one type to another? Answered
  from the store's own rows, never from a list in the code.
- **ALIGN(spec)** — which stored worked examples match this task's
  structure? A matching example may be reused whole (only if every
  distinctive word of the task is accounted for) or through a
  canonical reading of its code (an ascending loop that returns its
  first hit *is* the minimum of a set — a fact about Python, reusable
  with the reduction swapped when the task says "largest").
- **CONTRAST(examples)** — what property separates the examples that
  expect `True` from those that expect `False`? A separating property is
  only accepted if it is connected to the task's words; otherwise it is
  reported as a conjecture.

Grammar itself is knowledge here: the closed words of English —
quantifiers ("all", "any"), reciprocals ("each other"), negation, the
copula, "of"-chains, plurals, "count/number of" — live in the store as
`ENG:` rows, so "all the numbers are different from each other" compiles
to a pairwise check with no code written for it.

## Knowledge sources

All knowledge enters through one door: a source adapter proposes rows,
an admission check filters them, and nothing is ever *used* unless the
task's own oracle (tests, proofs) approves the result. Registered
sources today:

| source | answers | example |
|---|---|---|
| the knowledge store | word meanings, English facts | "prime" -> a primality test |
| WordNet | synonyms, antonyms | "largest" ~ opposite of "smallest" |
| the Python runtime | real callables by name | "repeat" -> `itertools.repeat` |
| worked-example corpora | whole solutions, code shapes | armed per run via `--calibrate` |
| registered documents | prose + inline code | python.org tutorial taught "substring = s[i:j]" |
| Wikidata | defining formulas of concepts | "woodall number" -> `n·2^n − 1` -> a checker |
| Wiktionary | whether two words are related | "sign" is defined using "negative" |
| OEIS | integer sequences by example values | Lucas numbers -> the recurrence, mined and run |
| Wikipedia | article text for a topic | source for digests (below) |

Network sources are cached, rate-limited, and fail silently to a normal
refusal. Background world knowledge stays opt-in for theory benchmarks
whose facts are deliberately fictional.

## Producing text-like artifacts: the digest

"Write a summary of the history of Ireland" is recognized by shape (an
instruction verb plus a text-artifact noun — summary, overview,
abstract...). The engine does not write prose. It retrieves the source
(or uses provided text), reads it into a graph of relations with the
source sentence attached to each, selects the most connected and
date-anchored relations, and returns the **summary graph** with three
checks: *faithfulness* (every selected relation exists in the source),
*compression* (how much was kept), and *round-trip* (re-reading the
extractive rendering recovers the graph). Turning the graph into fluent
prose is a job for a separate renderer — whose output this engine can
then verify by reading it back.

## Results (zero LLM calls everywhere)

| benchmark | result | wrong |
|---|---|---|
| LGP-20 + SP-6 grid/scheduling puzzles (vs gold) | 26/26 | 0 |
| ZebraLogicBench, all sizes (1000) | 959 certified, 41 excluded (encoding) | 0 |
| BBH logical_deduction 3/5/7 objects (750) | 750/750 | 0 |
| BBH web_of_lies (250) | 250/250 | 0 |
| ProofWriter depths 0-5 + variants (800) | 800/800; paraphrased split abstains | 0 |
| StepGame spatial (500) | 387 correct, 112 abstain | 1* |
| Story QA, 20 tasks (20,000) | 19,238 correct + 755 honest abstains | 7* |
| MBPP, store knowledge only | 37 solved, 7 proven-unsatisfiable | 0 |
| MBPP + worked-example corpora and live sources | further solves, graded per source, spot-audited | 0 found |
| AR-LSAT (230) | honest refusals — prose-inventory reading not built yet | 0 |

\* All 8 disagreements trace to documented label errors in the datasets
themselves (verifiable by hand), not to the engine.

## How honesty is enforced, concretely

- **Certificates.** A puzzle answer means "unique solution, proven by
  exhaustive search". A program means "assembled from parts the task's
  words license AND passes every test". A derived fact carries its
  proof chain.
- **Licenses** gate reuse before the tests even run: a worked example
  may only be replayed if it accounts for every distinctive word of the
  task (a prime-checker cannot answer a Woodall task, even though it
  passes the tests); a candidate must use every independently-varying
  argument; a property that separates the examples but connects to none
  of the task's words is a conjecture, not an answer.
- **Attribution.** "The manager claimed X" is stored as a fact about the
  manager, never as X.
- **Nothing is deleted.** Time-varying facts get interval stamps, so
  history stays queryable and contradictions are detected, not
  overwritten.
- **Quarantine.** Reference solutions found in the input are locked away
  unused.

## Repository layout

```
graphstep/
  unified.py        the six steps (one entry: solve)
  run.py            run any HuggingFace dataset or JSON file
  agenda.py         the solver loop: ground -> decompose -> question ->
                    retrieve -> derive -> verify -> learn -> refuse
  engine/
    core.py         finite-domain solver: propagation, search,
                    uniqueness certificates, unsat cores
    constraints.py  the constraint library
    ir.py           the logic intermediate representation
    sandbox.py      runs candidate programs against the task's tests
  reading/          ALL domain knowledge lives here as data
    kb.py           THE knowledge store: one relation of rows — word
                    meanings, English facts (ENG: rows), clue shapes,
                    code atoms; provenance on every row
    questions.py    the question layer + external sources (Wikidata,
                    Wiktionary, OEIS, Wikipedia, documents) + digests
    semgraph.py     clause reader: every clause captured with roles,
                    tense, attribution; two trust tiers
    narrative.py    time-stamped facts for stories ("where is the
                    violin now?")
    worldmodel.py   open-prose facts, rules, forward chaining with proofs
    compose.py      grammar-driven composition ("sum OF digits OF n"
                    IS sum(digits(n)))
    sources.py      the source/admission machinery + worked-example
                    corpora + runtime introspection
    retrieval.py    WordNet/Wikidata for the claim algebra
    kbstore.py      persistent cache of retrieved knowledge
  tests/            CI gates: generality (no task code), reading
                    honesty, puzzle parity vs gold, compiler neutrality
  legacy/           superseded scripts kept only as parity references
  results/          benchmark result files
  data/             runtime caches (not committed)
```

## Running it

```bash
pip install -r requirements.txt          # datasets, spacy, z3, nltk
python3 -m spacy download en_core_web_sm

# any dataset — the sample's own shape decides everything:
python3 -m graphstep.run --hf <org/dataset> [--split test] [--limit N]
    [--drop <answer fields>] [--loop]

# arm a worked-example corpus (any dataset of description/code pairs):
#   --calibrate "graphstep.reading.sources:register_procedure_corpus:\
#                <org/dataset>@<split>:<textfield>:<codefield>"
# register a prose document as a source:
#   --calibrate "graphstep.reading.questions:register_document:<path>|<name>"

# one sample from JSON / stdin:
python3 -m graphstep.unified sample.json

# the CI gates:
python3 graphstep/tests/test_generality.py
python3 graphstep/tests/test_unified_parity.py
python3 graphstep/tests/test_compose.py
python3 graphstep/tests/test_sources.py
python3 graphstep/tests/test_semgraph.py
```

## How the system grows

Everything grows by adding **rows** (knowledge) or a **general
mechanism** (a new kind of certificate) — never a branch:

1. A new phrasing for a known meaning -> one row, or free via WordNet.
2. A new word meaning -> usually mined from usages, formulas, or the
   runtime; hand rows are allowed but provenance-stamped and counted.
3. A new sentence shape -> one row with an emitter.
4. A new capability (a solver, an artifact type) -> a general module any
   input can trigger by its shape.

The metric that keeps this honest: **hand-authored rows per newly solved
sample must trend to zero** — solved tasks should increasingly be paid
for with retrieved and mined knowledge, not with engineering.

## Roadmap

- Mine the digest selection policy from Wikipedia (article body vs. its
  human-written lead section) instead of the current hand-set weights.
- AR-LSAT reading: prose inventories ("the eight students—George, ..."),
  selection ("exactly six will report"), nested unless-conditionals.
- Program synthesis growth: map/superlative devices from the store's
  English rows, more usage shapes ("pairs", "rotations"), richer
  introspection (methods of built-in types).
- Promote captured prose to proof grade in place (the reading frontier —
  the current limit on digest quality).
