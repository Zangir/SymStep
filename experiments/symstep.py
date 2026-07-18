#!/usr/bin/env python3
"""
SymStep: Symbolic Step Verification for LLM Logical Reasoning
NILA Workshop @ IJCAI-ECAI 2026 -- Experiment Runner

Methods compared:
  1. direct       -- zero-shot answer
  2. cot          -- chain-of-thought, then answer
  3. self_refine  -- generate → self-critique → refine (1 round)
  4. logic_lm     -- Logic-LM style: LLM writes Z3 Python code → execute
  5. symstep      -- [OURS] one deduction at a time, each verified by a
                      symbolic constraint propagator
  6. symstep_g    -- [OURS+] symstep + MRV guidance from symbolic state
"""

import subprocess, re, copy, time, json, os, sys, tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

def _find_claude_bin() -> str:
    """Locate the Claude Code CLI binary across common install locations."""
    if env := os.environ.get("CLAUDE_BIN"):
        return env
    import glob
    candidates = [
        os.path.expanduser("~/.claude/local/claude"),
        os.path.expanduser("~/.local/bin/claude"),
        "/usr/local/bin/claude",
        *glob.glob(os.path.expanduser(
            "~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude"
        )),
        *glob.glob(os.path.expanduser(
            "~/.cursor/extensions/anthropic.claude-code-*/resources/native-binary/claude"
        )),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return "claude"  # assume on PATH

CLAUDE_BIN = _find_claude_bin()
MODEL = os.environ.get("SYMSTEP_MODEL", "haiku")  # override: SYMSTEP_MODEL=sonnet python3 symstep.py

# ── Puzzle dataclass ──────────────────────────────────────────────────────────

@dataclass
class Puzzle:
    name: str
    difficulty: str          # easy | medium | hard
    people: List[str]
    attributes: Dict[str, List[str]]   # attr → possible values
    clues: List[str]
    solution: Dict[str, Dict[str, str]]  # person → attr → value

# ── Hand-crafted puzzles with verified solutions ──────────────────────────────

PUZZLES = [
    # ── EASY: 3 people, 2 attributes ─────────────────────────────────────────
    Puzzle(
        name="E1-Pets", difficulty="easy",
        people=["Alice", "Bob", "Carol"],
        attributes={
            "color": ["Red", "Blue", "Green"],
            "pet":   ["Cat", "Dog", "Fish"],
        },
        clues=[
            "Alice does not have a dog.",
            "Bob lives in the blue house.",
            "The person in the red house has a cat.",
            "Carol lives in the green house.",
            "Carol has a fish.",
        ],
        solution={
            "Alice": {"color": "Red",   "pet": "Cat"},
            "Bob":   {"color": "Blue",  "pet": "Dog"},
            "Carol": {"color": "Green", "pet": "Fish"},
        },
    ),
    Puzzle(
        name="E2-Jobs", difficulty="easy",
        people=["Adam", "Beth", "Chris"],
        attributes={
            "drink": ["Tea", "Coffee", "Water"],
            "job":   ["Doctor", "Teacher", "Engineer"],
        },
        clues=[
            "Adam drinks tea.",
            "The doctor drinks coffee.",
            "Beth is not the engineer.",
            "Chris is not the doctor.",
            "Adam is not the teacher.",
        ],
        solution={
            "Adam":  {"drink": "Tea",    "job": "Engineer"},
            "Beth":  {"drink": "Coffee", "job": "Doctor"},
            "Chris": {"drink": "Water",  "job": "Teacher"},
        },
    ),
    # ── MEDIUM: 4 people, 2 attributes ───────────────────────────────────────
    Puzzle(
        name="M1-Houses", difficulty="medium",
        people=["Alice", "Bob", "Carol", "Dave"],
        attributes={
            "color": ["Red", "Blue", "Green", "Yellow"],
            "drink": ["Tea", "Coffee", "Juice", "Water"],
        },
        clues=[
            "Alice drinks tea.",
            "Bob lives in the red house.",
            "The person in the blue house drinks coffee.",
            "Carol does not live in the green house.",
            "Dave drinks water.",
            "Alice does not live in the yellow house.",
        ],
        solution={
            "Alice": {"color": "Green",  "drink": "Tea"},
            "Bob":   {"color": "Red",    "drink": "Juice"},
            "Carol": {"color": "Blue",   "drink": "Coffee"},
            "Dave":  {"color": "Yellow", "drink": "Water"},
        },
    ),
    Puzzle(
        name="M2-Nationals", difficulty="medium",
        people=["Ana", "Ben", "Carl", "Donna"],
        attributes={
            "nationality": ["English", "French", "German", "Spanish"],
            "pet":         ["Cat", "Dog", "Bird", "Fish"],
        },
        clues=[
            "The English person has a cat.",
            "Ana is not French.",
            "Ben is German.",
            "The Spanish person has a bird.",
            "Carl does not have a dog.",
            "Donna is not English.",
            "Ben has a fish.",
            "Ana is not Spanish.",
        ],
        solution={
            "Ana":   {"nationality": "English", "pet": "Cat"},
            "Ben":   {"nationality": "German",  "pet": "Fish"},
            "Carl":  {"nationality": "Spanish", "pet": "Bird"},
            "Donna": {"nationality": "French",  "pet": "Dog"},
        },
    ),
    # ── HARD: 4 people, 3 attributes ─────────────────────────────────────────
    Puzzle(
        name="H1-Full", difficulty="hard",
        people=["Alex", "Blake", "Casey", "Drew"],
        attributes={
            "color": ["Red", "Blue", "Green", "Yellow"],
            "drink": ["Tea", "Coffee", "Juice", "Water"],
            "pet":   ["Cat", "Dog", "Bird", "Fish"],
        },
        clues=[
            "Alex drinks tea.",
            "Blake lives in the red house.",
            "The person in the blue house drinks coffee.",
            "The person in the green house has a bird.",
            "Casey does not have a cat.",
            "Drew drinks water.",
            "The person who drinks juice has a dog.",
            "Alex does not live in the yellow house.",
            "Casey lives in the blue house.",
        ],
        solution={
            "Alex":  {"color": "Green",  "drink": "Tea",    "pet": "Bird"},
            "Blake": {"color": "Red",    "drink": "Juice",  "pet": "Dog"},
            "Casey": {"color": "Blue",   "drink": "Coffee", "pet": "Fish"},
            "Drew":  {"color": "Yellow", "drink": "Water",  "pet": "Cat"},
        },
    ),
    Puzzle(
        name="H2-Full", difficulty="hard",
        people=["Pam", "Quinn", "Rosa", "Sam"],
        attributes={
            "color": ["Red", "Blue", "Green", "Yellow"],
            "job":   ["Doctor", "Teacher", "Engineer", "Chef"],
            "pet":   ["Cat", "Dog", "Bird", "Fish"],
        },
        clues=[
            "Pam is not the chef.",
            "Quinn lives in the blue house.",
            "The doctor has a cat.",
            "The person in the red house is a teacher.",
            "Rosa does not live in the green house.",
            "Sam is not the doctor.",
            "The engineer has a dog.",
            "Pam lives in the red house.",
            "Quinn has a bird.",
        ],
        solution={
            "Pam":   {"color": "Red",    "job": "Teacher",  "pet": "Fish"},
            "Quinn": {"color": "Blue",   "job": "Chef",     "pet": "Bird"},
            "Rosa":  {"color": "Yellow", "job": "Doctor",   "pet": "Cat"},
            "Sam":   {"color": "Green",  "job": "Engineer", "pet": "Dog"},
        },
    ),
]

# ── Scheduling Benchmark SP-6 ─────────────────────────────────────────────────
# Six scheduling/assignment puzzles across easy/medium/hard.
# Same bijective CSP structure as LGP-14 but a different domain (time slots,
# roles, labs, venues) to test cross-domain generalization.

SCHEDULING_PUZZLES = [
    # ── EASY: 3 people, 2 attributes ─────────────────────────────────────────
    Puzzle(
        name="SP-E1-Workshop", difficulty="easy",
        people=["Alex", "Beth", "Carlos"],
        attributes={
            "slot":  ["Morning", "Afternoon", "Evening"],
            "topic": ["Security", "Privacy", "Ethics"],
        },
        clues=[
            "Alex presents in the Morning session.",
            "The Security talk is in the Morning session.",
            "Beth does not present on Ethics.",
            "The Privacy talk is in the Evening session.",
            "Carlos does not present in the Morning session.",
        ],
        solution={
            "Alex":   {"slot": "Morning",   "topic": "Security"},
            "Beth":   {"slot": "Evening",   "topic": "Privacy"},
            "Carlos": {"slot": "Afternoon", "topic": "Ethics"},
        },
    ),
    Puzzle(
        name="SP-E2-Lab", difficulty="easy",
        people=["Dana", "Eva", "Felix"],
        attributes={
            "shift":     ["Day", "Night", "Weekend"],
            "equipment": ["Microscope", "Centrifuge", "Scanner"],
        },
        clues=[
            "Dana works the Day shift.",
            "The Centrifuge is used on the Night shift.",
            "Eva does not use the Microscope.",
            "The Scanner is used on the Weekend.",
            "Felix does not work the Day shift.",
            "Eva does not work the Night shift.",
        ],
        solution={
            "Dana":  {"shift": "Day",     "equipment": "Microscope"},
            "Eva":   {"shift": "Weekend", "equipment": "Scanner"},
            "Felix": {"shift": "Night",   "equipment": "Centrifuge"},
        },
    ),
    # ── MEDIUM: 4 people, 2 attributes ───────────────────────────────────────
    Puzzle(
        name="SP-M1-Projects", difficulty="medium",
        people=["Gina", "Hank", "Irene", "Jack"],
        attributes={
            "week": ["Week1", "Week2", "Week3", "Week4"],
            "role": ["Lead", "Dev", "QA", "Design"],
        },
        clues=[
            "Irene starts in Week 1.",
            "The Lead role begins in Week 2.",
            "Jack is the QA engineer.",
            "Hank does not start in Week 1.",
            "The Design phase is in Week 4.",
            "Gina does not work as Developer.",
            "Gina is not assigned to the Design phase.",
        ],
        solution={
            "Gina":  {"week": "Week2", "role": "Lead"},
            "Hank":  {"week": "Week4", "role": "Design"},
            "Irene": {"week": "Week1", "role": "Dev"},
            "Jack":  {"week": "Week3", "role": "QA"},
        },
    ),
    Puzzle(
        name="SP-M2-Research", difficulty="medium",
        people=["Kira", "Leo", "Mia", "Nate"],
        attributes={
            "lab":   ["Lab-A", "Lab-B", "Lab-C", "Lab-D"],
            "topic": ["CV", "NLP", "RL", "Theory"],
        },
        clues=[
            "Leo is assigned to Lab-A.",
            "The RL team is in Lab-B.",
            "Kira does not work on CV.",
            "The NLP team is in Lab-C.",
            "Mia does not work on NLP.",
            "Nate is not assigned to Lab-A.",
            "The Theory group is in Lab-D.",
            "Nate does not work on Theory.",
            "Nate is not assigned to Lab-C.",
        ],
        solution={
            "Kira": {"lab": "Lab-C", "topic": "NLP"},
            "Leo":  {"lab": "Lab-A", "topic": "CV"},
            "Mia":  {"lab": "Lab-D", "topic": "Theory"},
            "Nate": {"lab": "Lab-B", "topic": "RL"},
        },
    ),
    # ── HARD: 4 people, 3 attributes ─────────────────────────────────────────
    Puzzle(
        name="SP-H1-Department", difficulty="hard",
        people=["Owen", "Priya", "Quinn", "Rita"],
        attributes={
            "floor": ["1st", "2nd", "3rd", "4th"],
            "role":  ["Manager", "Engineer", "Designer", "Analyst"],
            "shift": ["Morning", "Afternoon", "Evening", "Night"],
        },
        clues=[
            "The Manager works on the 4th floor.",
            "Priya is not an Engineer.",
            "Owen works in the Afternoon.",
            "The Designer works the Night shift.",
            "Quinn is on the 1st floor.",
            "Rita does not work the Morning shift.",
            "The Analyst works the Evening shift.",
            "Owen does not work on the 1st floor.",
            "Priya works the Morning shift.",
            "Quinn does not work as Designer.",
            "The Engineer works on the 2nd floor.",
        ],
        solution={
            "Owen":  {"floor": "2nd", "role": "Engineer", "shift": "Afternoon"},
            "Priya": {"floor": "4th", "role": "Manager",  "shift": "Morning"},
            "Quinn": {"floor": "1st", "role": "Analyst",  "shift": "Evening"},
            "Rita":  {"floor": "3rd", "role": "Designer", "shift": "Night"},
        },
    ),
    Puzzle(
        name="SP-H2-EventCrew", difficulty="hard",
        people=["Sam", "Tara", "Uma", "Vince"],
        attributes={
            "venue": ["Hall-A", "Hall-B", "Hall-C", "Hall-D"],
            "task":  ["Setup", "Catering", "Tech", "Security"],
            "time":  ["8am", "10am", "12pm", "2pm"],
        },
        clues=[
            "Uma is assigned to Hall-A.",
            "The Setup crew works at 8am.",
            "The Tech team is in Hall-C.",
            "Vince works at 12pm.",
            "The Security team works at 2pm.",
            "Sam does not work on Security.",
            "The Catering team works at 10am.",
            "The Catering team is in Hall-B.",
            "The Setup team is in Hall-A.",
        ],
        solution={
            "Sam":   {"venue": "Hall-B", "task": "Catering", "time": "10am"},
            "Tara":  {"venue": "Hall-D", "task": "Security", "time": "2pm"},
            "Uma":   {"venue": "Hall-A", "task": "Setup",    "time": "8am"},
            "Vince": {"venue": "Hall-C", "task": "Tech",     "time": "12pm"},
        },
    ),
]

# ── Constraint Propagation Verifier ──────────────────────────────────────────

class ConstraintPropagator:
    """Maintains a possibility matrix and enforces arc-consistency."""

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        self.poss: Dict[str, Dict[str, set]] = {
            p: {a: set(vals) for a, vals in puzzle.attributes.items()}
            for p in puzzle.people
        }

    def apply_positive(self, person: str, attr: str, value: str) -> Tuple[bool, str]:
        """Assert person[attr] == value."""
        if person not in self.poss:
            return False, f"Unknown person '{person}'"
        if attr not in self.poss[person]:
            return False, f"Unknown attribute '{attr}'"
        canon_val = self._match_value(attr, value)
        if canon_val is None:
            return False, f"Unknown value '{value}' for attribute '{attr}'"
        if canon_val not in self.poss[person][attr]:
            return False, (
                f"Contradiction: {person} cannot have {attr}={canon_val} "
                f"(already eliminated; possibilities: {sorted(self.poss[person][attr])})"
            )
        self.poss[person][attr] = {canon_val}
        for other in self.puzzle.people:
            if other != person:
                self.poss[other][attr].discard(canon_val)
                if not self.poss[other][attr]:
                    return False, (
                        f"Contradiction: assigning {canon_val} to {person} "
                        f"leaves {other}.{attr} with no possibilities"
                    )
        self._propagate()
        return True, "OK"

    def apply_negative(self, person: str, attr: str, value: str) -> Tuple[bool, str]:
        """Assert person[attr] != value."""
        if person not in self.poss or attr not in self.poss[person]:
            return False, f"Unknown person/attr: {person}.{attr}"
        canon_val = self._match_value(attr, value)
        if canon_val is None:
            return False, f"Unknown value '{value}' for attribute '{attr}'"
        if canon_val not in self.poss[person][attr]:
            return True, "OK (already eliminated)"
        self.poss[person][attr].discard(canon_val)
        if not self.poss[person][attr]:
            return False, (
                f"Contradiction: eliminating {canon_val} from {person}.{attr} "
                "leaves no possibilities"
            )
        self._propagate()
        return True, "OK"

    def _match_value(self, attr: str, value: str) -> Optional[str]:
        """Case-insensitive match against known values."""
        for v in self.puzzle.attributes[attr]:
            if v.lower() == value.strip().lower():
                return v
        return None

    def _propagate(self):
        """Simple unit propagation: if a value is uniquely assigned, remove from others."""
        changed = True
        while changed:
            changed = False
            for attr in self.puzzle.attributes:
                for person in self.puzzle.people:
                    if len(self.poss[person][attr]) == 1:
                        val = next(iter(self.poss[person][attr]))
                        for other in self.puzzle.people:
                            if other != person and val in self.poss[other][attr]:
                                self.poss[other][attr].discard(val)
                                changed = True

    def is_solved(self) -> bool:
        return all(len(self.poss[p][a]) == 1
                   for p in self.puzzle.people
                   for a in self.puzzle.attributes)

    def get_solution(self) -> Optional[Dict[str, Dict[str, str]]]:
        if not self.is_solved():
            return None
        return {p: {a: next(iter(self.poss[p][a])) for a in self.puzzle.attributes}
                for p in self.puzzle.people}

    def state_summary(self) -> str:
        lines = []
        for p in self.puzzle.people:
            parts = []
            for a in self.puzzle.attributes:
                vs = sorted(self.poss[p][a])
                parts.append(f"{a}={'|'.join(vs)}" if len(vs) > 1 else f"{a}={vs[0]}")
            lines.append(f"  {p}: {', '.join(parts)}")
        return "\n".join(lines)

    def guidance(self) -> str:
        """Point the LLM at the most constrained undetermined (person, attr) pair."""
        best, best_n = None, float("inf")
        for p in self.puzzle.people:
            for a in self.puzzle.attributes:
                n = len(self.poss[p][a])
                if 1 < n < best_n:
                    best, best_n = (p, a, sorted(self.poss[p][a])), n
        if best:
            p, a, vs = best
            return f"[Hint] {p}'s {a} must be one of: {{{', '.join(vs)}}}"
        return "[All values determined]"

# ── LLM helper ───────────────────────────────────────────────────────────────

def call_llm(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            result = subprocess.run(
                [CLAUDE_BIN, "--print", "--model", MODEL, "--no-session-persistence"],
                input=prompt, capture_output=True, text=True, timeout=180
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"    [timeout, retrying in {wait}s...]", flush=True)
                time.sleep(wait)
            else:
                return "[TIMEOUT]"
    return "[TIMEOUT]"

# ── Shared formatting ─────────────────────────────────────────────────────────

def puzzle_text(p: Puzzle) -> str:
    attrs = ", ".join(f"{a} ({'/'.join(v)})" for a, v in p.attributes.items())
    clues = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(p.clues))
    people = ", ".join(p.people)
    return (
        f"LOGIC PUZZLE\n"
        f"People: {people}\n"
        f"Attributes: {attrs}\n"
        f"Clues:\n{clues}"
    )

def answer_format(p: Puzzle) -> str:
    return (
        "Provide your final answer as:\n"
        + "\n".join(f"  {person}: " + ", ".join(f"{a}=<value>" for a in p.attributes)
                    for person in p.people)
    )

def parse_solution(text: str, p: Puzzle) -> Dict[str, Dict[str, str]]:
    """Extract person→attr→value from LLM output.

    Robust to verbose/concise outputs (Haiku and Sonnet).  Strategy:
    0. Answer-block lines: parse each person's attributes only from the line
       that *begins* with that person's name (ignoring markdown emphasis such
       as ``**Alice:**``), so verbose reasoning cannot bleed one person's
       values onto another. This is the canonical answer format and the most
       reliable signal for both concise (Haiku) and verbose (Sonnet) outputs.
    1. For each person, find ALL occurrences of their name in the text.
    2. For each occurrence, search a 500-char window for attr=value patterns.
    3. Also accept natural-language patterns: "person is|has|: value".
    4. Fallback: search the full text for "PersonName … attr … Value" proximity.
    """
    result: Dict[str, Dict[str, str]] = {person: {} for person in p.people}

    # --- Strategy 0: structured answer block, parsed line-by-line ----------
    # A line such as "Alice: color=Red, pet=Cat" or "**Bob:** color=Blue" is
    # anchored to a single person; we extract attr=value pairs from that line
    # ONLY. Later answer lines overwrite earlier ones (the final block wins).
    name_anchor = {
        person: re.compile(rf"^[|*#>_\s\-]*{re.escape(person)}\b[*_:\s\-]*", re.IGNORECASE)
        for person in p.people
    }
    for line in text.splitlines():
        for person in p.people:
            m = name_anchor[person].match(line)
            if not m:
                continue
            rest = line[m.end():]
            for attr, vals in p.attributes.items():
                val_lower = {v.lower(): v for v in vals}
                am = re.search(
                    rf"(?i)\b{re.escape(attr)}\s*[=:\-]?\s*([A-Za-z][A-Za-z0-9_\-]*)",
                    rest,
                )
                if am:
                    raw = am.group(1).strip().rstrip(".,;)")
                    if raw.lower() in val_lower:
                        result[person][attr] = val_lower[raw.lower()]
            break  # this line is anchored to one person; don't test others

    for person in p.people:
        # Collect all starting positions where this person's name appears.
        person_positions = [m.start() for m in re.finditer(
            rf"(?i)\b{re.escape(person)}\b", text
        )]
        if not person_positions:
            continue

        for attr, vals in p.attributes.items():
            if attr in result[person]:
                continue  # already resolved by the answer-block strategy
            val_lower = {v.lower(): v for v in vals}
            found = False

            # --- Strategy 1: scan 500-char window around each mention of person ---
            for pos in person_positions:
                # Look both before (50 chars) and after (500 chars) the name
                snippet = text[max(0, pos - 50): pos + 500]
                # Patterns: attr=Val, attr: Val, attr is Val, attr - Val
                patterns = [
                    rf"(?i)\b{re.escape(attr)}\s*[=:\-]\s*([A-Za-z][A-Za-z0-9_\-]*)",
                    rf"(?i)\b{re.escape(attr)}\s+is\s+([A-Za-z][A-Za-z0-9_\-]*)",
                    rf"(?i)\b{re.escape(attr)}\s+(?:of\s+)?([A-Za-z][A-Za-z0-9_\-]*)",
                ]
                for pat in patterns:
                    m = re.search(pat, snippet)
                    if m:
                        raw = m.group(1).strip().rstrip(".,;)")
                        if raw.lower() in val_lower:
                            result[person][attr] = val_lower[raw.lower()]
                            found = True
                            break
                if found:
                    break

            if found:
                continue

            # --- Strategy 2: look for any value directly adjacent to person name ---
            for pos in person_positions:
                snippet = text[max(0, pos - 20): pos + 300]
                for v in vals:
                    if re.search(rf"(?i)\b{re.escape(v)}\b", snippet):
                        # Make sure no other person's name intervenes
                        intervening = False
                        for other in p.people:
                            if other != person:
                                other_m = re.search(
                                    rf"(?i)\b{re.escape(other)}\b", snippet
                                )
                                if other_m:
                                    v_m = re.search(rf"(?i)\b{re.escape(v)}\b", snippet)
                                    if v_m and other_m.start() < v_m.start():
                                        intervening = True
                                        break
                        if not intervening:
                            result[person][attr] = v
                            found = True
                            break
                if found:
                    break

            if found:
                continue

            # --- Strategy 3: full-text search for attr=Val near person ---
            for v in vals:
                pattern_global = rf"(?i)\b{re.escape(person)}\b[^.!?\n]{{0,200}}\b{re.escape(v)}\b"
                if re.search(pattern_global, text):
                    result[person][attr] = v
                    break

    return result

def check_solution(got: Dict[str, Dict[str, str]], expected: Dict[str, Dict[str, str]]) -> bool:
    for person, attrs in expected.items():
        for attr, val in attrs.items():
            if got.get(person, {}).get(attr) != val:
                return False
    return True

# ── Logic-LM (Z3 code generation) baseline ───────────────────────────────────

LOGIC_LM_PROMPT_TMPL = """\
You are an expert Z3 constraint programmer. Solve this logic puzzle by writing \
complete, executable Python code using the z3 library.

{puzzle_text}

Write Python code that:
1. `from z3 import *`
2. Creates an EnumSort for each attribute listing ALL its possible values.
3. For each person creates a z3 Const of the appropriate EnumSort per attribute.
4. Adds "all-different" constraints (each attribute value assigned to exactly one person).
5. Encodes EVERY clue above as z3 constraints (using == and != on the Consts).
6. Creates a Solver, adds all constraints, calls solver.check().
7. If sat, prints the model in this exact format (one line per person):
   PersonName: attr1=Value1, attr2=Value2
   (use the exact person names and value names from the puzzle)

Output ONLY the Python code — no markdown, no explanations."""


def _extract_code(text: str) -> str:
    m = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1)
    return text.strip()


def _safe_exec(code: str, timeout: int = 20) -> str:
    """Execute Python code in a subprocess; return stdout or error tag."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        r = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout
        )
        return r.stdout if r.returncode == 0 else f"[ERROR] {r.stderr[:300]}"
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[EXEC-ERROR] {e}"
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def _parse_code_output(stdout: str, p: Puzzle) -> Dict[str, Dict[str, str]]:
    """Parse 'PersonName: attr=val, attr=val' lines into a solution dict."""
    sol: Dict[str, Dict[str, str]] = {}
    for line in stdout.splitlines():
        for person in p.people:
            if line.strip().lower().startswith(person.lower()):
                sol[person] = {}
                for attr, vals in p.attributes.items():
                    pat = rf"(?i){re.escape(attr)}\s*[=:]\s*(\S+)"
                    m = re.search(pat, line)
                    if m:
                        raw = m.group(1).rstrip(",")
                        for v in vals:
                            if v.lower() == raw.lower():
                                sol[person][attr] = v
                                break
                break
    return sol


def run_logic_lm(p: Puzzle) -> Tuple[bool, int, float, int]:
    """Logic-LM: ask LLM to write Z3 solver code, execute it, verify solution."""
    t0 = time.time()
    prompt = LOGIC_LM_PROMPT_TMPL.format(puzzle_text=puzzle_text(p))
    response = call_llm(prompt)
    code = _extract_code(response)
    stdout = _safe_exec(code, timeout=20)
    sol = _parse_code_output(stdout, p)
    correct = check_solution(sol, p.solution) if sol else False
    return correct, 1, time.time() - t0, 0


# ── Method 1: Direct ─────────────────────────────────────────────────────────

def run_direct(p: Puzzle) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    prompt = puzzle_text(p) + "\n\n" + answer_format(p)
    response = call_llm(prompt)
    got = parse_solution(response, p)
    correct = check_solution(got, p.solution)
    return correct, 1, time.time() - t0, 0

# ── Method 2: Chain-of-Thought ────────────────────────────────────────────────

def run_cot(p: Puzzle, save_trace: bool = False) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    prompt = (
        puzzle_text(p)
        + "\n\nThink step by step through the clues to determine each person's attributes. "
        + "Show your reasoning, then give the final answer.\n\n"
        + answer_format(p)
    )
    response = call_llm(prompt)
    got = parse_solution(response, p)
    correct = check_solution(got, p.solution)
    if save_trace and not correct:
        trace_file = f"cot_failure_{p.name.replace('-','_').lower()}.txt"
        with open(trace_file, "w") as f:
            f.write(f"=== CoT FAILURE TRACE: {p.name} ===\n\n")
            f.write("PROMPT:\n" + prompt + "\n\n")
            f.write("RESPONSE:\n" + response + "\n\n")
            f.write(f"PARSED: {got}\n")
            f.write(f"EXPECTED: {p.solution}\n")
        print(f"    [trace saved → {trace_file}]")
    return correct, 1, time.time() - t0, 0

# ── Method 3: Self-Refine ─────────────────────────────────────────────────────

def run_self_refine(p: Puzzle) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    prompt1 = (
        puzzle_text(p)
        + "\n\nSolve the puzzle and provide your answer.\n\n"
        + answer_format(p)
    )
    answer1 = call_llm(prompt1)

    prompt2 = (
        puzzle_text(p)
        + "\n\nYour previous answer was:\n"
        + answer1
        + "\n\nNow carefully verify each clue against your answer. "
        "List any violated clues, correct them, and provide the revised final answer.\n\n"
        + answer_format(p)
    )
    answer2 = call_llm(prompt2)
    got = parse_solution(answer2, p)
    correct = check_solution(got, p.solution)
    return correct, 2, time.time() - t0, 0

# ── Method 4 & 5: SymStep (our novel methods) ─────────────────────────────────

SYMSTEP_SYSTEM = """\
You are solving a logic puzzle by making one deduction at a time.
Each deduction must follow EXACTLY one of these formats:
  DEDUCE: <Person>, <attribute>, <Value>        (person HAS this value)
  DEDUCE: <Person>, <attribute>, NOT <Value>    (person does NOT have this value)
After each deduction you will receive verification feedback.
When the puzzle is fully solved, output:  CONCLUDE: done
Do not output multiple deductions at once."""

def run_symstep(p: Puzzle, with_guidance: bool = False,
                max_steps: int = 25) -> Tuple[bool, int, float, int]:
    t0 = time.time()
    prop = ConstraintPropagator(p)
    calls = 0
    contradictions = 0
    history: List[str] = []

    intro = (
        puzzle_text(p)
        + "\n\n"
        + SYMSTEP_SYSTEM
        + "\n\nMake your first deduction now."
    )
    history.append(intro)

    for _ in range(max_steps):
        prompt = "\n\n".join(history)
        response = call_llm(prompt)
        calls += 1
        history.append(f"[You]: {response}")

        if re.search(r"CONCLUDE\s*:\s*done", response, re.IGNORECASE):
            break

        m = re.search(
            r"DEDUCE\s*:\s*([A-Za-z]\w*)\s*,\s*([A-Za-z]\w*)\s*,\s*(NOT\s+)?([A-Za-z0-9][A-Za-z0-9_\-]*)",
            response, re.IGNORECASE
        )
        if not m:
            history.append("[Verifier]: Could not parse your deduction. Use exactly:\n"
                           "  DEDUCE: <Person>, <attribute>, <Value>  OR\n"
                           "  DEDUCE: <Person>, <attribute>, NOT <Value>")
            continue

        person    = m.group(1).strip()
        attr      = m.group(2).strip().lower()
        is_neg    = m.group(3) is not None
        value     = m.group(4).strip()
        attr_norm = next((a for a in p.attributes if a.lower() == attr), attr)

        if is_neg:
            ok, msg = prop.apply_negative(person, attr_norm, value)
        else:
            ok, msg = prop.apply_positive(person, attr_norm, value)

        if ok:
            feedback = f"[Verifier]: ✓ Correct deduction accepted. {msg}"
            if prop.is_solved():
                feedback += "\n[Verifier]: Puzzle fully solved! Output: CONCLUDE: done"
            elif with_guidance:
                feedback += "\n" + prop.guidance()
        else:
            contradictions += 1
            feedback = (
                f"[Verifier]: ✗ CONTRADICTION detected — {msg}\n"
                "Please reconsider and make a different deduction."
            )

        history.append(feedback)

        if prop.is_solved():
            break

    sol = prop.get_solution() or {}
    correct = check_solution(sol, p.solution)
    return correct, calls, time.time() - t0, contradictions

# ── Experiment runner ─────────────────────────────────────────────────────────

METHODS = {
    "direct":      run_direct,
    "cot":         run_cot,
    "self_refine": run_self_refine,
    "logic_lm":    run_logic_lm,
    "symstep":     lambda p: run_symstep(p, with_guidance=False),
    "symstep_g":   lambda p: run_symstep(p, with_guidance=True),
}

# Subset without logic_lm for backward-compatible runs
METHODS_NO_LLM = {k: v for k, v in METHODS.items() if k != "logic_lm"}

def run_all(puzzles=None, out_file="results.json"):
    if puzzles is None:
        puzzles = PUZZLES
    print("=" * 70)
    print(f"SymStep Experiment — model={MODEL}  puzzles={len(puzzles)}")
    print("=" * 70)

    results: Dict[str, Dict] = {
        m: {"correct": 0, "total": 0, "calls": 0, "contradictions": 0, "time": 0.0}
        for m in METHODS
    }
    per_puzzle = []

    for puzzle in puzzles:
        print(f"\n── Puzzle: {puzzle.name} ({puzzle.difficulty}) ──")
        row = {"puzzle": puzzle.name, "difficulty": puzzle.difficulty}
        for method_name, method_fn in METHODS.items():
            result = method_fn(puzzle)
            correct, calls, elapsed = result[0], result[1], result[2]
            contra = result[3] if len(result) > 3 else 0
            results[method_name]["correct"]        += int(correct)
            results[method_name]["total"]          += 1
            results[method_name]["calls"]          += calls
            results[method_name]["contradictions"] += contra
            results[method_name]["time"]           += elapsed
            print(f"  {method_name:12s}  {'✓' if correct else '✗'}  "
                  f"calls={calls}  contra={contra}  t={elapsed:.1f}s")
            row[method_name] = {"correct": correct, "calls": calls,
                                "contradictions": contra, "time": round(elapsed, 1)}
        per_puzzle.append(row)

    print("\n" + "=" * 70)
    print(f"{'Method':<14} {'Acc':>6} {'Avg calls':>10} {'Avg contra':>12} {'Avg time':>10}")
    print("-" * 70)
    for m, r in results.items():
        acc    = r["correct"]        / r["total"] * 100
        acalls = r["calls"]          / r["total"]
        acont  = r["contradictions"] / r["total"]
        atime  = r["time"]           / r["total"]
        print(f"{m:<14} {acc:>5.0f}%  {acalls:>9.1f}  {acont:>11.1f}  {atime:>8.1f}s")
    print("=" * 70)

    out = {"model": MODEL, "summary": results, "per_puzzle": per_puzzle}
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_file}")

    print("\nAccuracy by difficulty:")
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in per_puzzle if r["difficulty"] == diff]
        if not subset:
            continue
        print(f"\n  {diff.upper()} ({len(subset)} puzzles):")
        for m in METHODS:
            n_correct = sum(1 for r in subset if r[m]["correct"])
            print(f"    {m:<14} {n_correct}/{len(subset)}")

if __name__ == "__main__":
    run_all()
