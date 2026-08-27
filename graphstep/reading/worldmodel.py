#!/usr/bin/env python3
"""General world models: arbitrary text -> abstract logic -> deduction.

Every sentence is read into one of a few ABSTRACT logical forms — nothing
domain-specific lives in the machinery:

  Literal   fact about the world:  (polarity, subject, predicate, object?)
            "Charlie is not quiet" -> (-, charlie, quiet, None)
            "the dog needs the bear" -> (+, dog, need, bear)
  Rule      quantified implication: "If someone is rough and they need the
            bear then they chase the dog" -> conditions + conclusion with a
            shared variable ("someone/they").  Generic plurals are rules
            too: "Wolves are afraid of mice."
  ScaleEdge ordering/offset on an induced scale: comparatives ("bigger
            than") create a scale from their own stem; a declarative LEXICON
            maps phrase families (spatial words, clock faces, compass) onto
            abstract scales with signed deltas.  The scales themselves are
            just integer axes to the engine.

The engine side:
  closure()      forward-chains rules to a fixpoint -> NEW derived facts,
                 each with its proof chain (deducing more logic).
  ask(statement) True / False / Unknown by open-world entailment.
  vector(a, b)   forced signed offset between two things on the scales,
                 via Offset propagation (multi-hop composition).

Zero LLM calls anywhere in this module.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

VAR_WORDS = {"someone", "something", "somebody", "they", "it", "he", "she"}
IRREGULAR_PLURAL = {"people": "person", "mice": "mouse", "wolves": "wolf",
                    "geese": "goose", "children": "child", "sheep": "sheep",
                    "men": "man", "women": "woman", "things": "thing"}
GENERIC_SORTALS = {"person", "thing", "one", "body", "individual"}


def norm_ent(s: str) -> str:
    s = re.sub(r"^(the|a|an)\s+", "", s.strip().lower())
    return re.sub(r"\s+", "_", s.strip(" .,"))


class Literal:
    __slots__ = ("pos", "subj", "pred", "obj", "origin")

    def __init__(self, pos: bool, subj: str, pred: str, obj: Optional[str],
                 origin: Optional[str] = None):
        self.pos, self.subj, self.pred, self.obj = pos, subj, pred, obj
        self.origin = origin

    def key(self):
        return (self.subj, self.pred, self.obj)

    def signed(self):
        return (self.pos, *self.key())

    def subst(self, var_val) -> "Literal":
        if isinstance(var_val, dict):
            s = var_val.get(self.subj, self.subj)
            o = var_val.get(self.obj, self.obj) if self.obj else None
            return Literal(self.pos, s, self.pred, o)
        s = var_val if self.subj == "?" else self.subj
        o = var_val if self.obj == "?" else self.obj
        return Literal(self.pos, s, self.pred, o)

    def variables(self):
        vs = set()
        for t in (self.subj, self.obj):
            if t and t.startswith("?"):
                vs.add(t)
        return vs

    def __repr__(self):
        n = "" if self.pos else "NOT "
        return f"{n}{self.pred}({self.subj}{', ' + self.obj if self.obj else ''})"


class Rule:
    def __init__(self, conds: List[Literal], concl: Literal, origin: str):
        self.conds, self.concl, self.origin = conds, concl, origin

    def __repr__(self):
        return " & ".join(map(repr, self.conds)) + " -> " + repr(self.concl)


# ------------------------------------------------------------------ lexicons
# Declarative world knowledge: phrase family -> signed deltas on named
# abstract scales. The engine knows nothing about "space"; these are just
# scales like any comparative's.
SPATIAL_LEXICON = [
    (r"upper[- ]left|top[- ]left|north ?west|10 o'clock|11 o'clock", (-1, 1)),
    (r"upper[- ]right|top[- ]right|north ?east|1 o'clock|2 o'clock", (1, 1)),
    (r"lower[- ]left|bottom[- ]left|south ?west|7 o'clock|8 o'clock", (-1, -1)),
    (r"lower[- ]right|bottom[- ]right|south ?east|4 o'clock|5 o'clock", (1, -1)),
    (r"\bleft\b|\bwest\b|9 o'clock", (-1, 0)),
    (r"\bright\b|\beast\b|3 o'clock", (1, 0)),
    (r"\babove\b|\bover\b|\btop\b|\bnorth\b|\bup\b|12 o'clock|\bhigher\b",
     (0, 1)),
    (r"\bbelow\b|\bunder\b|\bbottom\b|\bsouth\b|\bdown\b|6 o'clock|\blower\b",
     (0, -1)),
]
OVERLAP_RE = re.compile(r"overlap|same location|same position|coincide", re.I)

# calibrated template dictionary: signature -> (dx, dy) meaning
# "first-mentioned name is at (dx, dy) relative to second-mentioned name".
# Populated from a benchmark's VALIDATION split (learned world knowledge);
# the heuristic axis-scan below is the fallback for unseen phrasings.
TEMPLATE_VECTORS: dict = {}

# direction lexicon: forced (sign dx, sign dy) -> the English direction word
LABEL_OF = {(-1, 0): "left", (1, 0): "right", (0, 1): "above",
            (0, -1): "below", (-1, 1): "upper-left", (1, 1): "upper-right",
            (-1, -1): "lower-left", (1, -1): "lower-right",
            (0, 0): "overlap"}


def calibrate_spatial_templates(source: str) -> None:
    """Learn template -> displacement-vector rows from a LABELED source
    given as "dataset@split" (fields: story, question, label). An item
    whose story has exactly ONE direction-bearing sentence reveals that
    template's meaning through its gold label. Only templates consistent
    across >=90%% of their votes are admitted; test data is never touched."""
    import re as _re
    from collections import defaultdict as _dd
    from datasets import load_dataset
    name, split = source.split("@")
    val = load_dataset(name, split=split)
    vec_of = {v: k for k, v in LABEL_OF.items()}
    votes = _dd(lambda: _dd(int))
    for x in val:
        sents = [s for s in _re.split(r"(?<=[.!?])\s+",
                                      str(x["story"]).strip()) if s.strip()]
        bearing = [s for s in sents
                   if _axis_vector(s) != (0, 0)
                   or _re.search(r"o'?clock|clock|corner|degree", s, _re.I)]
        if len(bearing) != 1:
            continue
        sent = bearing[0]
        names = list(dict.fromkeys(_re.findall(r"\b([A-Z])\b", sent)))
        qm = _re.search(r"agent (\w+) to the agent (\w+)",
                        str(x["question"]))
        if len(names) != 2 or qm is None or x["label"] not in vec_of:
            continue
        g = vec_of[x["label"]]
        if (qm.group(1), qm.group(2)) == (names[0], names[1]):
            v = g
        elif (qm.group(1), qm.group(2)) == (names[1], names[0]):
            v = (-g[0], -g[1])
        else:
            continue
        votes[spatial_signature(sent, names)][v] += 1
    for sig, vs in votes.items():
        best = max(vs, key=vs.get)
        if vs[best] >= 0.9 * sum(vs.values()):
            TEMPLATE_VECTORS[sig] = best
    print(f"  calibrated {len(TEMPLATE_VECTORS)} spatial templates "
          f"from {source}")

X_WORDS = [(r"\bleft\b|\bwest\b|\blefthand\b", -1),
           (r"\bright\b|\beast\b|\brighthand\b", 1)]
Y_WORDS = [(r"\babove\b|\bover\b(?!\s+there)|\btop\b|\bnorth\b|\bup\b"
            r"|\bupper\b|\bhigher\b", 1),
           (r"\bbelow\b|\bunder\b|\bbottom\b|\bsouth\b|\bdown\b"
            r"|\blower\b", -1)]
CLOCK_VEC = {12: (0, 1), 1: (1, 1), 2: (1, 1), 3: (1, 0), 4: (1, -1),
             5: (1, -1), 6: (0, -1), 7: (-1, -1), 8: (-1, -1), 9: (-1, 0),
             10: (-1, 1), 11: (-1, 1)}


def spatial_signature(sent: str, names) -> str:
    """Mask entity names (in mention order) and strip cosmetic variation so
    surface templates align; direction words are preserved untouched."""
    sig = sent.strip().rstrip(".")
    for i, n in enumerate(names):
        sig = re.sub(rf"\b{re.escape(n)}\b", f"<{i}>", sig)
    sig = sig.lower()
    sig = re.sub(r"[,.;!]", " ", sig)
    sig = re.sub(r"\b(the|a|an|object|objects|agent|slightly|quite|too)\b",
                 " ", sig)
    return re.sub(r"\s+", " ", sig).strip()


def _axis_vector(text: str):
    """Independent per-axis scan of one clause -> (dx, dy)."""
    dx = dy = 0
    m = re.search(r"(\d{1,2})\s*(?:o'?clock|position of a clock|"
                  r"on a clock)", text, re.I)
    if m and int(m.group(1)) in CLOCK_VEC:
        return CLOCK_VEC[int(m.group(1))]
    for pat, v in X_WORDS:
        if re.search(pat, text, re.I):
            dx = v
            break
    for pat, v in Y_WORDS:
        if re.search(pat, text, re.I):
            dy = v
            break
    return (dx, dy)

# antonym pairs for one-dimensional comparatives: both directions live on one
# induced scale ("big": bigger -> +, smaller -> -)
ANTONYM_SCALES = [
    ("big", "small"), ("large", "little"), ("old", "young"), ("old", "new"),
    ("fast", "slow"), ("heavy", "light"), ("tall", "short"),
    ("high", "low"), ("expensive", "cheap"), ("strong", "weak"),
    ("early", "late"), ("hot", "cold"), ("wide", "narrow"),
]


def comparative_scale(word: str) -> Optional[Tuple[str, int]]:
    """'bigger' -> ('big', +1); 'smaller' -> ('big', -1); None otherwise."""
    w = word.lower()
    if not (w.endswith("er") or w.startswith("more ")):
        return None
    stem = w[5:] if w.startswith("more ") else w[:-2]
    stem = stem[:-1] if stem.endswith(("gg", "nn", "tt"), 0) else stem
    for a, b in ANTONYM_SCALES:
        if stem.startswith(a[:max(3, len(a) - 1)]):
            return (a, 1)
        if stem.startswith(b[:max(3, len(b) - 1)]):
            return (a, -1)
    return (stem, 1)                       # unknown comparative: own scale


# ------------------------------------------------------------------ parsing
IRREGULAR_VERB = {"has": "have", "does": "do", "goes": "go", "is": "be",
                  "are": "be", "was": "be", "were": "be"}


def norm_verb(v: str) -> str:
    """chases/needs/has -> chase/need/have (shared by all parsers)."""
    v = v.lower()
    if v in IRREGULAR_VERB:
        return IRREGULAR_VERB[v]
    if v.endswith("ies"):
        return v[:-3] + "y"
    if v.endswith("s") and not v.endswith("ss"):
        return v[:-1]
    return v


def norm_pred_noun(w: str) -> str:
    """Singularize plural-noun predicates so 'primates' == 'primate'.
    Adjective endings (-ous, -ss, -is, -us) are left alone."""
    wl = w.lower()
    if wl in IRREGULAR_PLURAL:
        return IRREGULAR_PLURAL[wl]
    if wl.endswith("ies") and len(wl) > 4:
        return wl[:-3] + "y"
    if (wl.endswith("s") and len(wl) > 3
            and not wl.endswith(("ss", "ous", "us", "is", "os"))):
        return wl[:-1]
    return wl


RULE_VARS = {"a", "b", "c", "x", "y", "z"}


def _mk_subj(raw: str) -> str:
    s = norm_ent(raw)
    if s in VAR_WORDS:
        return "?"
    if s in RULE_VARS:
        return "?" + s          # single-letter rule variable ("If A ... B")
    return s


DISCOURSE_RE = re.compile(
    r"^(also|however|then|thus|therefore|moreover|furthermore|still|so|"
    r"in addition|in fact|of course|besides)[,\s]+", re.I)


def parse_literal(text: str) -> Optional[Literal]:
    """One simple clause -> Literal. '?'-subject for someone/they."""
    tl = DISCOURSE_RE.sub("", text.strip().rstrip(".").strip().lower())

    # negated verb: "<S> does not <verb> [the] <O>"
    m = re.match(r"(.+?)\s+(?:does not|do not|doesn't|don't|cannot)\s+"
                 r"(\w+)\s+(?:the |a |an )?(.+)$", tl)
    if m:
        return Literal(False, _mk_subj(m.group(1)), norm_verb(m.group(2)),
                       _mk_subj(m.group(3)))

    # copula: "<S> is [not] [a] <ADJ|TYPE>[ of <O>]" — the predicate must be
    # short and clause-free; messy prose falls through to the spaCy tier
    m = re.match(r"(.+?)\s+(?:is|are)\s+(not\s+)?(?:a |an )?(.+)$", tl)
    if m:
        rest = m.group(3).strip()
        clean = ("," not in rest and len(rest.split()) <= 4
                 and not re.search(r"\b(and|but|when|if|then|who|that|"
                                   r"which|because|also|yet)\b", rest))
        if clean:
            pos = m.group(2) is None
            subj = _mk_subj(m.group(1))
            m2 = re.match(r"(?:the\s+|a\s+|an\s+)?([\w\-]+)\s+"
                          r"(of|to|in|from)\s+(?:the\s+)?(.+)$", rest)
            if m2:                   # "afraid of wolves", "endemic to X"
                return Literal(pos, subj, f"{m2.group(1)}_{m2.group(2)}",
                               _mk_subj(m2.group(3)))
            if len(rest.split()) <= 2:
                return Literal(pos, subj,
                               norm_pred_noun(norm_ent(rest)), None)

    COPULAR = {"is", "are", "was", "were", "be", "been", "being"}

    def _clean_np(s: str) -> bool:
        return ("," not in s and len(s.split()) <= 4
                and not re.search(r"\b(is|are|was|were|and|that|which|who)\b",
                                  s))

    # positive verb with marked object: "<S> <verb>(s) the <O>"
    m = re.match(r"(.+?)\s+(\w+)\s+(?:the|a|an)\s+(.+)$", tl)
    if m and m.group(2) not in COPULAR and _clean_np(m.group(1)) \
            and _clean_np(m.group(3)):
        return Literal(True, _mk_subj(m.group(1)), norm_verb(m.group(2)),
                       _mk_subj(m.group(3)))
    # 3rd-person verb with bare object: "the circuit has electricity"
    m = re.match(r"(.+?)\s+(\w+s)\s+([\w\- ]{1,30})$", tl)
    if m and m.group(2) not in COPULAR and _clean_np(m.group(1)) \
            and _clean_np(m.group(3)):
        return Literal(True, _mk_subj(m.group(1)), norm_verb(m.group(2)),
                       _mk_subj(m.group(3)))
    return None


def parse_rule(sent: str) -> Optional[Rule]:
    rules = parse_rules(sent)
    return rules[0] if rules else None


def parse_rules(sent: str) -> List[Rule]:
    """All rules a sentence yields ('Lemurs are primates and prosimians'
    -> two rules)."""
    _extra_concls: List[str] = []
    tl = sent.strip().rstrip(".")
    m = re.match(r"[Ii]f\s+(.+?),?\s+then\s+(.+)$", tl)
    if m:
        conds = []
        prev_subj = "?"
        for c in re.split(r"\s+and\s+(?=[\w'])", m.group(1)):
            lit = parse_literal(c)
            if lit is None:
                m2 = re.match(r"(not\s+)?([\w\-]+)$", c.strip())
                if m2:                       # bare adjective conjunct
                    lit = Literal(m2.group(1) is None, prev_subj,
                                  m2.group(2).lower(), None)
            if lit is None:
                return None
            prev_subj = lit.subj
            conds.append(lit)
        concl = parse_literal(m.group(2))
        if conds and concl:
            return [Rule(conds, concl, sent.strip())]
        return []
    # generic plural: "Wolves are afraid of mice." / "All red things are
    # big." / "All red, kind people are white."
    m = re.match(r"(?:[Aa]ll\s+)?([\w\-, ]+?)\s+are\s+(not\s+)?(.+)$", tl)
    if m:
        subj_words = [w.strip() for w in
                      re.split(r",\s*|\s+and\s+|\s+", m.group(1)) if w.strip()]
        if not subj_words:
            return None
        sortal = subj_words[-1].lower()
        adjectives = [w.lower() for w in subj_words[:-1]]
        if sortal in VAR_WORDS or any(a in VAR_WORDS for a in adjectives):
            return None
        singular = IRREGULAR_PLURAL.get(
            sortal, sortal[:-1] if sortal.endswith("s") else None)
        if singular is None:
            return None                     # subject is not a plural class
        conds: List[Literal] = [Literal(True, "?", a, None)
                                for a in adjectives]
        if singular not in GENERIC_SORTALS:
            conds.append(Literal(True, "?", singular, None))
        if not conds:
            return None
        rest = m.group(3).strip()
        if " and " in rest and "," not in rest:
            parts = [p.strip() for p in rest.split(" and ")]
            if all(len(p.split()) <= 2 for p in parts):
                rest = parts[0]              # first conjunct here...
                _extra_concls.extend(parts[1:])   # ...the rest emitted below
        if "," in rest or len(rest.split()) > 5:
            return None                     # globbed conclusion: stay unread
        pos = m.group(2) is None
        m2 = re.match(r"(?:a |an )?([\w\-]+)\s+(of|to|in|from)\s+"
                      r"(?:the\s+)?(.+)$", rest)
        if m2 and len(m2.group(3).split()) <= 3:
            concl = Literal(pos, "?", f"{m2.group(1)}_{m2.group(2)}",
                            norm_ent(m2.group(3)))
        elif len(rest.split()) <= 2:
            concl = Literal(pos, "?",
                            norm_pred_noun(norm_ent(
                                re.sub(r"^(a|an)\s+", "", rest))), None)
        else:
            return None
        out = [Rule(conds, concl, sent.strip())]
        for extra in _extra_concls:
            out.append(Rule(conds,
                            Literal(pos, "?",
                                    norm_pred_noun(norm_ent(
                                        re.sub(r"^(a|an)\s+", "", extra))),
                                    None), sent.strip()))
        return out
    # generic plural with verb: "Metal things conduct electricity."
    NOT_SORTAL = {"this", "these", "those", "its", "his", "hers", "is",
                  "was", "has", "does", "as", "less", "thus", "various",
                  "perhaps", "across"}
    QUANTIFIERS = {"all", "every", "most", "some", "many", "these", "those",
                   "this", "such"}
    words = tl.replace(",", " ").split()
    if len(words) >= 2 and not tl.lower().startswith(("if ", "the ")):
        FUNCTION_WORDS = {"is", "are", "was", "were", "the", "a", "an",
                          "of", "to", "in", "and", "or", "not", "also"}
        for i, w in enumerate(words):
            wl = w.lower()
            if wl in NOT_SORTAL or "'" in wl or not wl.isalpha():
                continue
            if any(x.lower() in FUNCTION_WORDS or not x.isalpha()
                   for x in words[:i]):
                continue                     # function words never modify a sortal
            singular = IRREGULAR_PLURAL.get(
                wl, wl[:-1] if wl.endswith("s") and len(wl) > 3 else None)
            if singular is None or i == len(words) - 1:
                continue
            nxt = words[i + 1].lower()
            if nxt in ("is", "are", "was", "were", "and", "or"):
                break                        # copular/conjoined: not this form
            adjectives = [x.lower() for x in words[:i]
                          if x.lower() not in QUANTIFIERS]
            if any(a in VAR_WORDS for a in adjectives):
                break
            if all(x.lower() in QUANTIFIERS for x in words[:i]) and i == 0:
                pass
            # a proper noun before the "plural" word means this is a normal
            # sentence, not a generic statement ("Fred resembles ...")
            if any(w2[:1].isupper() and j > 0
                   for j, w2 in enumerate(sent.split()[:i])):
                break
            rest = " ".join(words[i + 1:])
            m3 = re.match(r"(?:do not\s+|don't\s+)?(\w+)\s+"
                          r"(?:the\s+|a\s+|an\s+)?(.+)$", rest)
            if not m3:
                break
            if (m3.group(1) in ("the", "a", "an", "is", "are", "was", "were")
                    or "," in m3.group(2) or len(m3.group(2).split()) > 3):
                break                        # not a clean generic-verb rule
            conds = [Literal(True, "?", a, None) for a in adjectives]
            if singular not in GENERIC_SORTALS:
                conds.append(Literal(True, "?", singular, None))
            if not conds:
                break
            obj = re.sub(r"\s+to\s+\w+$", "", m3.group(2))   # "water to survive"
            pos = "do not" not in rest and "don't" not in rest
            return [Rule(conds,
                         Literal(pos, "?", norm_verb(m3.group(1)),
                                 norm_ent(obj)), sent.strip())]
    return []


def parse_facts_spacy(sent: str) -> List["Literal"]:
    """Library-based (spaCy dependency) fact extraction for open prose.
    Reads MULTIPLE facts per sentence: conjoined predicates ("green, red
    and blue"), hedges ("seems to be", "is said to be", "often"), pronoun
    subjects resolved to the sentence's proper-noun subject, negation from
    the parse tree. Fallback tier: used only when the template parsers
    cannot read a sentence."""
    from .syntax_tier import _nlp
    doc = _nlp()(sent)
    out: List[Literal] = []
    proper = next((t.text.lower() for t in doc
                   if t.pos_ == "PROPN" and t.dep_ in
                   ("nsubj", "nsubjpass", "poss")), None)

    def subject_of(tok) -> Optional[str]:
        cur = tok
        for _ in range(6):                    # climb to the governing verb
            subj = next((c for c in cur.children
                         if c.dep_ in ("nsubj", "nsubjpass")), None)
            if subj is not None:
                if subj.pos_ == "PRON":
                    return proper
                # full span (handles hyphenated names split by the tokenizer)
                end = subj.i
                d = subj.doc
                while (end + 2 < len(d) and d[end + 1].text == "-"
                       and d[end + 2].is_alpha):
                    end += 2
                span = d[subj.left_edge.i: end + 1]
                return norm_ent(span.text.lower())
            if cur.head is cur:
                break
            cur = cur.head
        return None

    def negated(tok) -> bool:
        cur = tok
        for _ in range(4):
            if any(c.dep_ == "neg" for c in cur.children):
                return True
            if cur.head is cur:
                break
            cur = cur.head
        return False

    COND_MARKS = {"if", "when", "whenever", "unless", "while"}

    def in_subordinate(tok) -> bool:
        """Conditional clauses and relatives state hypotheses, not facts —
        but reason/elaboration clauses ("as", "because") do assert."""
        cur = tok
        for _ in range(8):
            if cur.dep_ in ("relcl", "acl", "csubj"):
                return True
            if cur.dep_ == "advcl":
                intro = {t.lower_ for t in cur.subtree if t.i < cur.i}
                if intro & COND_MARKS:
                    return True
            if cur.head is cur:
                return False
            cur = cur.head
        return False

    def _ok_subj(subj) -> bool:
        if subj is None or subj in VAR_WORDS:
            return False
        last = subj.split("_")[-1]            # "young_people" -> "people"
        return (last not in GENERIC_SORTALS
                and last not in IRREGULAR_PLURAL
                and not last.endswith("s"))   # generic plural = RULE, not fact

    for tok in doc:
        is_adj = (tok.pos_ == "ADJ" and tok.dep_ in ("acomp", "attr", "conj",
                                                     "oprd", "ccomp"))
        is_type = (tok.pos_ in ("NOUN", "PROPN")
                   and (tok.dep_ == "attr"
                        or (tok.dep_ == "conj" and tok.head.dep_ == "attr"))
                   and not any(c.dep_ == "prep" for c in tok.children))
        if not (is_adj or is_type):
            continue
        if in_subordinate(tok):
            continue
        subj = subject_of(tok)
        if not _ok_subj(subj):
            continue
        # "afraid of wolves": adjectival predicate with prep complement
        prep = next((c for c in tok.children if c.dep_ == "prep"), None)
        pobj = (next((g for g in prep.children if g.dep_ == "pobj"), None)
                if prep is not None else None)
        if is_adj and prep is not None and pobj is not None:
            out.append(Literal(not negated(tok), norm_ent(subj),
                               f"{tok.lemma_.lower()}_{prep.lower_}",
                               norm_ent(pobj.doc[
                                   pobj.left_edge.i: pobj.i + 1].text)))
            continue
        out.append(Literal(not negated(tok), norm_ent(subj),
                           norm_pred_noun(tok.lemma_.lower()), None))
        if is_type:
            # modifier decomposition: "a living thing" -> thing(x), living(x)
            for mod in tok.children:
                if mod.dep_ == "amod" and mod.pos_ in ("ADJ", "VERB"):
                    out.append(Literal(not negated(tok), norm_ent(subj),
                                       mod.lower_, None))   # surface form

    # verb facts: ROOT/main verbs with a direct object, or with a
    # prepositional complement ("was born in Paris" -> born_in)
    for tok in doc:
        if tok.pos_ != "VERB" or in_subordinate(tok):
            continue
        subj = subject_of(tok)
        if not _ok_subj(subj):
            continue
        dobj = next((c for c in tok.children if c.dep_ == "dobj"), None)
        if dobj is not None:
            obj_span = dobj.doc[dobj.left_edge.i: dobj.i + 1].text
            out.append(Literal(not negated(tok), norm_ent(subj),
                               norm_verb(tok.lemma_.lower()),
                               norm_ent(obj_span)))
            continue
        prep = next((c for c in tok.children if c.dep_ == "prep"), None)
        pobj = (next((g for g in prep.children if g.dep_ == "pobj"), None)
                if prep is not None else None)
        if pobj is not None and pobj.pos_ in ("PROPN", "NOUN"):
            vlem = (tok.lower_ if tok.tag_ == "VBN"
                    else norm_verb(tok.lemma_.lower()))
            out.append(Literal(not negated(tok), norm_ent(subj),
                               f"{vlem}_{prep.lower_}",
                               norm_ent(pobj.doc[
                                   pobj.left_edge.i: pobj.i + 1].text)))
    # dedupe, keep order
    seen, uniq = set(), []
    for l in out:
        if l.signed() not in seen:
            seen.add(l.signed())
            uniq.append(l)
    return uniq


# ------------------------------------------------------------------ the model
class WorldModel:
    def __init__(self):
        self.facts: Dict[tuple, Tuple[bool, str]] = {}   # key -> (pol, proof)
        self.rules: List[Rule] = []
        self.entities: set = set()
        self.scale_edges: List[Tuple[str, str, int, int, str]] = []
        self.overlaps: List[Tuple[str, str]] = []
        self.unread: List[str] = []
        self.disagreements: List[tuple] = []
        self.contradictions: List[tuple] = []
        self.fact_tier: Dict[tuple, str] = {}
        self.quantities: Dict[tuple, tuple] = {}   # (subj, attr) -> (mag, unit, src)
        self.induced_rules: List[str] = []
        self.retrieved_rules: List[Rule] = []
        self.retrieval_log: List[str] = []

    TIER_RANK = {"agree": 5, "compat": 4, "spacy": 3, "regex": 2,
                 "retrieved": 1, "derived": 0}

    def commit(self, lit: "Literal", sentence: str, tier: str):
        k = lit.key()
        if k in self.facts and self.facts[k][0] != lit.pos:
            # stated contradiction: resolve by reader-confidence tier,
            # record the episode either way
            old_tier = self.fact_tier.get(k, "regex")
            winner = (lit.pos if self.TIER_RANK.get(tier, 0) >
                      self.TIER_RANK.get(old_tier, 0) else self.facts[k][0])
            self.contradictions.append(
                (f"{'' if lit.pos else 'NOT '}{lit.pred}({lit.subj})",
                 f"conflicts with stored [{old_tier}]: {self.facts[k][1]}",
                 f"resolved by tier -> kept "
                 f"{'new' if winner == lit.pos else 'stored'}"))
            if winner == lit.pos:
                self.facts[k] = (lit.pos, sentence.strip())
                self.fact_tier[k] = tier
            return
        self.facts.setdefault(k, (lit.pos, sentence.strip()))
        self.fact_tier.setdefault(k, tier)
        self.entities.add(lit.subj)
        if lit.obj and lit.obj != "?":
            self.entities.add(lit.obj)

    def add_quantity(self, subj, attr, mag, unit, src):
        key = (subj, attr)
        if key in self.quantities:
            old = self.quantities[key]
            if abs(old[0] - mag) > 0.01 * max(abs(old[0]), abs(mag), 1e-9):
                self.contradictions.append(
                    (f"{attr}({subj}) = {mag} {unit}",
                     f"conflicts with stored {old[0]} {old[1]}: {old[2]}",
                     "kept stored (first assertion)"))
            return
        self.quantities[key] = (mag, unit, src.strip())
        self.entities.add(subj)

    # -------------------------------- construction
    @classmethod
    def from_text(cls, text: str,
                  spatial_entity_re: str = r"\b([A-Z][A-Za-z0-9]*)\b",
                  resolve_coref: bool = True):
        wm = cls()
        if resolve_coref:
            try:
                text = resolve_coreference(text)
            except Exception:
                pass
        sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        for s in (x.strip() for x in sentences if x.strip()):
            if wm._try_scale(s, spatial_entity_re):
                continue
            if re.search(r"\d", s) and _axis_vector(s) == (0, 0) \
                    and not re.search(r"o'?clock|degree|corner", s, re.I):
                try:
                    q = parse_quantity(s)
                except Exception:
                    q = None
                if q:
                    wm.add_quantity(*q, s)
                    continue
            rs = parse_rules(s)
            if rs:
                wm.rules.extend(rs)
                for r in rs:
                    for c in r.conds + [r.concl]:
                        if c.subj != "?":
                            wm.entities.add(c.subj)
                        if c.obj and c.obj != "?":
                            wm.entities.add(c.obj)
                continue
            # dual-reader fact commit: the regex reader PROPOSES, the
            # spaCy reader must AGREE (same signed literal) before a
            # regex fact commits. Disagreement is recorded, never guessed.
            lit = parse_literal(s)
            spacy_facts = []
            try:
                spacy_facts = parse_facts_spacy(s)
            except Exception:
                pass
            spacy_keys = {f.signed() for f in spacy_facts}
            if lit and lit.subj != "?":
                if lit.signed() in spacy_keys:
                    wm.commit(lit, s, tier="agree")
                    for f in spacy_facts:            # extra conjunct facts
                        if f.signed() != lit.signed():
                            wm.commit(f, s, tier="spacy")
                    continue
                if not spacy_facts:
                    # spaCy read nothing: commit regex proposal alone but
                    # tag it — single-source facts are auditable
                    wm.commit(lit, s, tier="regex")
                    continue
                def _head(p):
                    return p.split("_")[-1] if "_" in p else p

                compatible = [
                    f for f in spacy_facts
                    if f.subj == lit.subj and f.pos == lit.pos
                    and (_head(f.pred) == _head(lit.pred)
                         or f.pred == _head(lit.pred)
                         or lit.pred.startswith(f.pred + "_")
                         or f.pred.startswith(lit.pred + "_"))]
                if compatible:
                    wm.commit(lit, s, tier="compat")
                    for f in compatible:
                        wm.commit(f, s, tier="compat")
                    continue
                wm.disagreements.append(
                    (s.strip(), repr(lit), [repr(f) for f in spacy_facts]))
                wm.unread.append(s)
                continue
            if spacy_facts:
                for f in spacy_facts:
                    wm.commit(f, s, tier="spacy")
                continue
            wm.unread.append(s)
        return wm

    def _try_scale(self, sent: str, ent_re: str) -> bool:
        """Spatial/comparative sentences -> signed edges on abstract scales."""
        names = re.findall(ent_re, sent)
        names = [n for n in names if len(n) <= 3 and n.isupper()] or None
        # comparatives: "The X is bigger than the Y."
        m = re.match(r"[Tt]he (.+?) is (\w+er|more \w+) than the (.+?)\.?$",
                     sent.strip())
        if m:
            sc = comparative_scale(m.group(2))
            if sc:
                scale, sign = sc
                self.scale_edges.append(
                    (norm_ent(m.group(1)), norm_ent(m.group(3)),
                     0 if scale != "x" else sign, sign, sent.strip()))
                # generic 1-D scale: reuse y-slot for the magnitude axis
                self.scale_edges[-1] = (norm_ent(m.group(1)),
                                        norm_ent(m.group(3)), sign, 0,
                                        sent.strip())
                return True
        if names and len(set(names)) == 1 and len(names) >= 2 \
                and _axis_vector(sent) != (0, 0):
            # degenerate self-relation ("A is above A"): contradictory
            # noise — recorded, consumed, never blocks the rest
            self.contradictions.append(
                (f"self-relation on {names[0]}", sent.strip(),
                 "vacuous/contradictory: ignored"))
            return True
        if names and len(names) >= 2:
            a, b = names[0], names[1]
            if OVERLAP_RE.search(sent):
                self.overlaps.append((a, b))
                return True
            sig = spatial_signature(sent, [a, b])
            if sig in TEMPLATE_VECTORS:              # calibrated template
                dx, dy = TEMPLATE_VECTORS[sig]
                if (dx, dy) == (0, 0):
                    self.overlaps.append((a, b))
                else:
                    self.scale_edges.append((a, b, dx, dy, sent.strip()))
                return True
            # Unseen direction-bearing template: ABSTAIN (record as unread)
            # rather than guess — precision over coverage.
            if _axis_vector(sent) != (0, 0):
                return False
            # direction-free filler ("X and Y are over there"): no edge —
            # but only for short copular sentences with no logical content
            if (len(sent.split()) <= 8
                    and re.search(r"\bare\b|\bis\b", sent)
                    and not re.search(r"\b(if|then|of|not)\b", sent, re.I)):
                return True
        return False

    # -------------------------------- deduction (closure with proofs)
    def closure(self, max_rounds: int = 60, include_induced: bool = False):
        """Forward-chain rules to fixpoint. Join-based grounding: bindings
        are enumerated from the facts that actually match each condition
        (indexed by predicate), not from the product of all entities.
        Derived facts carry tiers: min(premise tiers, rule tier) — anything
        touched by an induced rule is tier 'induced' and can only ever
        license "Likely" answers. Returns [(literal, proof)] of new facts."""
        derived = []
        rule_sets = [(self.rules, "derived")]
        if getattr(self, "retrieved_rules", None):
            rule_sets.append((self.retrieved_rules, "retrieved"))
        if include_induced and getattr(self, "induced_impl", None):
            rule_sets.append((self.induced_impl, "induced"))

        for _ in range(max_rounds):
            changed = False
            # per-round fact indexes (facts only grow within a round set)
            pos_idx: Dict[tuple, list] = {}
            neg_idx: Dict[tuple, list] = {}
            pos_by_subj: Dict[tuple, list] = {}
            neg_by_subj: Dict[tuple, list] = {}
            for (s, p, o), (pos, _) in self.facts.items():
                idx, bys = ((pos_idx, pos_by_subj) if pos
                            else (neg_idx, neg_by_subj))
                idx.setdefault((p, o is None), []).append((s, o))
                bys.setdefault((p, o is None, s), []).append((s, o))

            def match_bindings(conds, binding, idx=0):
                """Backtracking join over conditions using the indexes."""
                if idx == len(conds):
                    yield dict(binding)
                    return
                lit = conds[idx]
                unary = lit.obj is None
                # bound subject -> indexed lookup instead of a table scan
                subj_bound = (lit.subj if not (lit.subj or "").startswith("?")
                              else binding.get(lit.subj))
                if subj_bound is not None:
                    table = pos_by_subj if lit.pos else neg_by_subj
                    rows = table.get((lit.pred, unary, subj_bound), ())
                else:
                    table = pos_idx if lit.pos else neg_idx
                    rows = table.get((lit.pred, unary), ())
                for s, o in rows:
                    add = {}
                    ok = True
                    for var, val in ((lit.subj, s), (lit.obj, o)):
                        if var is None or not var.startswith("?"):
                            if var is not None and var != val:
                                ok = False
                                break
                        elif binding.get(var, val) != val:
                            ok = False
                            break
                        else:
                            add[var] = val
                    if not ok:
                        continue
                    binding.update(add)
                    yield from match_bindings(conds, binding, idx + 1)
                    for k in add:
                        binding.pop(k, None)

            for rules, rule_tier in rule_sets:
                for rule in rules:
                    concl_vars = rule.concl.variables()
                    cond_vars = set().union(*[c.variables()
                                              for c in rule.conds]) \
                        if rule.conds else set()
                    if not concl_vars <= cond_vars:
                        continue          # a conclusion may not invent entities
                    for binding in match_bindings(rule.conds, {}):
                        concl = rule.concl.subst(binding)
                        k = concl.key()
                        premise_keys = [c.subst(binding).key()
                                        for c in rule.conds]
                        if k in self.facts:
                            if self.facts[k][0] != concl.pos:
                                self.contradictions.append(
                                    (repr(concl),
                                     f"stated: {self.facts[k][1]}",
                                     f"derived via: {rule.origin} [because "
                                     + "; ".join(self.facts[pk][1]
                                                 for pk in premise_keys
                                                 if pk in self.facts) + "]"))
                            continue
                        proof = (f"{rule.origin}  [because "
                                 + "; ".join(self.facts[pk][1]
                                             for pk in premise_keys
                                             if pk in self.facts) + "]")
                        self.facts[k] = (concl.pos, proof)
                        WEAK = {"induced": 0, "retrieved": 1, "derived": 2}
                        tier = rule_tier
                        for pk in premise_keys:
                            pt = self.fact_tier.get(pk)
                            if pt in WEAK and WEAK[pt] < WEAK.get(tier, 2):
                                tier = pt
                        self.fact_tier[k] = tier
                        row = (k[0], k[2])
                        tgt = (pos_idx, pos_by_subj) if concl.pos else \
                            (neg_idx, neg_by_subj)
                        tgt[0].setdefault((k[1], k[2] is None),
                                          []).append(row)
                        tgt[1].setdefault((k[1], k[2] is None, k[0]),
                                          []).append(row)
                        derived.append((concl, proof))
                        changed = True
            if not changed:
                break
        return derived

    def _holds(self, lit: Literal) -> bool:
        hit = self.facts.get(lit.key())
        return hit is not None and hit[0] == lit.pos

    # ------------------------- rule induction (mine -> verify -> tier)
    def induce(self, min_support: int = 3, assume_closed: bool = False):
        """Mine implication rules and exclusions from the current facts,
        verify them, and store them as tier-'induced' rules. They fire only
        via closure(include_induced=True) and can only license "Likely"
        answers — certified True/False answers are untouched."""
        from .induction import mine_rules, mine_exclusions, verify_rules
        mined = mine_rules(self, min_support, assume_closed)
        kept, retractions = verify_rules(self, mined)
        self.induced_impl = kept
        for r in retractions:
            self.contradictions.append((r, "", "retracted before use"))
        for a, b, n in mine_exclusions(self, min_support):
            self.induced_rules.append(
                f"[induced] {a} and {b} are mutually exclusive "
                f"(disjoint over {n} subjects)")
        self.induced_rules.extend(r.origin for r in kept)
        return kept

    # ------------------------- retrieval-augmented deduction
    def retrieve_for(self, query: str, sources=("wordnet",),
                     max_terms: int = 6, depth: int = 2,
                     link_entities: bool = False):
        """Close the query's knowledge gaps from external KBs. Admitted
        knowledge lands as tier-'retrieved' rules; the answer vocabulary
        stays True/False — provenance travels in proofs and ask_explained."""
        from .retrieval import find_gaps, SOURCES, admit
        gaps = find_gaps(self, query, depth,
                         link_entities=link_entities)[:max_terms]
        candidates = []
        for term in gaps:
            for s in sources:
                candidates.extend(SOURCES[s](term))
        rule_cands = [c for c in candidates if isinstance(c, Rule)]
        fact_cands = [c for c in candidates if isinstance(c, Literal)]
        admitted, rejected = admit(self, rule_cands)
        self.retrieved_rules.extend(admitted)
        for f in fact_cands:                  # facts: text authority via tiers
            self.commit(f, f.origin or f"[retrieved] {f.pred}({f.subj})",
                        tier="retrieved")
        self.retrieval_log.extend(
            [f"gap terms: {gaps}"] + [r.origin for r in admitted] + rejected)
        for rej in rejected:
            self.contradictions.append((rej, "", "text authority upheld"))
        return admitted

    def ask_explained(self, statement: str) -> dict:
        """Verdict + mandatory provenance: tier, proof, and the knowledge
        sources the proof rests on."""
        verdict = self.ask(statement)
        lit = parse_literal(statement)
        out = {"verdict": verdict, "tier": None, "proof": None, "sources": []}
        if lit is not None and lit.key() in self.facts:
            out["tier"] = self.fact_tier.get(lit.key(), "stated")
            out["proof"] = self.facts[lit.key()][1]
            out["sources"] = sorted(set(
                re.findall(r"\[retrieved:(\w+)", out["proof"] or "")
                + (["induced"] if "[induced" in (out["proof"] or "") else [])))
        return out

    # ------------------------- induced integrity rules (rules FROM facts)
    def induce_integrity(self):
        """Deduce general integrity rules from the world's own regularities
        and check them: (1) a binary predicate that is single-valued for
        every observed subject (>=3 subjects) is flagged FUNCTIONAL — a
        second value for any subject then reads as a contradiction;
        (2) antonym-pair attributes (tall/short, hot/cold ...) may not both
        hold positively of one subject."""
        by_pred: Dict[str, Dict[str, set]] = {}
        for (s, p, o), (pos, _) in self.facts.items():
            if o is not None and pos:
                by_pred.setdefault(p, {}).setdefault(s, set()).add(o)
        for p, subj_map in by_pred.items():
            if len(subj_map) >= 3 and all(len(v) == 1
                                          for v in subj_map.values()):
                self.induced_rules.append(
                    f"[induced] {p} is single-valued "
                    f"(every one of {len(subj_map)} subjects has exactly "
                    f"one {p})")
            multi = {s: v for s, v in subj_map.items() if len(v) > 1}
            singles = sum(1 for v in subj_map.values() if len(v) == 1)
            if multi and singles >= 3:
                for s, vals in multi.items():
                    self.contradictions.append(
                        (f"{p}({s}) has {len(vals)} values {sorted(vals)}",
                         f"but {p} is single-valued for {singles} other "
                         f"subjects", "flagged (induced functionality)"))
        for a, b in ANTONYM_SCALES:
            for (s, p, o), (pos, src1) in list(self.facts.items()):
                if p == a and o is None and pos:
                    hit = self.facts.get((s, b, None))
                    if hit and hit[0]:
                        self.contradictions.append(
                            (f"{a}({s}) and {b}({s})",
                             f"antonyms both asserted: {src1} / {hit[1]}",
                             "flagged (antonym exclusion)"))
        return self.induced_rules

    # -------------------------------- self-probe (write/read round trip)
    def render(self, key: tuple, pos: bool) -> str:
        """Literal -> canonical English, for round-trip probing."""
        subj, pred, obj = key
        s = subj.replace("_", " ")
        neg = "" if pos else " not"
        if obj is None:
            return f"{s} is{neg} a {pred.replace('_', ' ')}"
        m = re.match(r"(.+)_(of|to|in|from)$", pred)
        if m:
            return (f"{s} is{neg} {m.group(1)} {m.group(2)} "
                    f"{obj.replace('_', ' ')}")
        if pos:
            return f"{s} {pred}s the {obj.replace('_', ' ')}"
        return f"{s} does not {pred} the {obj.replace('_', ' ')}"

    def self_probe(self) -> List[str]:
        """Every stored fact, rendered back to English and re-asked, must
        answer True. A failure means the writer and the reader of the fact
        store disagree (normalization drift) — the silent-mismatch bug
        class. Returns the failing renderings."""
        failures = []
        for key, (pos, _) in list(self.facts.items()):
            probe = self.render(key, pos)
            if self.ask(probe) != "True":
                failures.append(probe)
        return failures

    COMP_ATTR = {"long": "length", "tall": "height", "high": "height",
                 "heavy": "mass", "old": "age", "big": "size",
                 "wide": "width", "deep": "depth", "fast": "speed"}

    # -------------------------------- queries
    def ask(self, statement: str) -> str:
        m = re.match(r"(?:is |are )?(?:the )?(.+?) (\w+er|more \w+) than "
                     r"(?:the )?(.+?)\??$", statement.strip().lower())
        if m:
            sc = comparative_scale(m.group(2))
            if sc:
                stem, sign = sc
                attr = self.COMP_ATTR.get(stem, stem)
                qa = self.quantities.get((norm_ent(m.group(1)), attr))
                qb = self.quantities.get((norm_ent(m.group(3)), attr))
                if qa and qb and qa[1] == qb[1]:
                    diff = qa[0] - qb[0]
                    if diff == 0:
                        return "False"
                    return "True" if (diff > 0) == (sign > 0) else "False"
            return "Unknown"
        return self._ask_literal(statement)

    def _ask_literal(self, statement: str) -> str:
        """Open-world entailment. True/False only from certified (stated or
        stated-rule-derived) facts; anything resting on an induced rule is
        gated to Likely/LikelyNot."""
        lit = parse_literal(statement)
        if lit is None:
            return "Unknown"
        hit = self.facts.get(lit.key())
        if hit is None:
            return "Unknown"
        if self.fact_tier.get(lit.key()) == "induced":
            return "Likely" if hit[0] == lit.pos else "LikelyNot"
        return "True" if hit[0] == lit.pos else "False"

    def wh_ask(self, subj: str, pred: str) -> Optional[str]:
        """'What is <subj> <pred>-of?' over derived facts."""
        for (s, p, o), (pos, _) in self.facts.items():
            if pos and s == subj and p == pred and o:
                return o
        return None

    def _connected(self, a: str, b: str) -> bool:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from([a, b])
        g.add_edges_from((s, o) for s, o, *_ in self.scale_edges)
        g.add_edges_from(self.overlaps)
        return nx.has_path(g, a, b)

    def vector(self, a: str, b: str,
               metric: bool = True) -> Optional[Tuple[int, int]]:
        """Forced (sign dx, sign dy) of a relative to b on the scales, via
        Z3 linear integer arithmetic (multi-hop composition by an existing
        SMT solver rather than a hand-rolled propagator).

        metric=True : edges are exact unit displacements (equations).
        metric=False: edges are qualitative orderings (inequalities); each
                      axis is reported only when ENTAILED (unsat of the
                      negation), else None.
        """
        import z3
        if not self._connected(a, b):
            return None
        names = {a, b} | {x for x, y, *_ in self.scale_edges} \
            | {y for x, y, *_ in self.scale_edges} \
            | {x for pair in self.overlaps for x in pair}
        X = {n: z3.Int(f"x_{n}") for n in names}
        Y = {n: z3.Int(f"y_{n}") for n in names}
        base = []
        for s, o, dx, dy, _ in self.scale_edges:
            for V, d in ((X, dx), (Y, dy)):
                if metric:
                    base.append(V[s] == V[o] + d)
                elif d > 0:
                    base.append(V[s] > V[o])
                elif d < 0:
                    base.append(V[s] < V[o])
        for s, o in self.overlaps:
            base.append(X[s] == X[o])
            base.append(Y[s] == Y[o])

        s0 = z3.Solver()
        s0.add(base)
        if s0.check() == z3.unsat:
            self.contradictions.append(
                ("scale constraints are mutually inconsistent",
                 "; ".join(e[4] for e in self.scale_edges)[:200], ""))
            return None

        def entailed_sign(va, vb) -> Optional[int]:
            solver = z3.Solver()
            solver.add(base)
            if solver.check(va >= vb) == z3.unsat:
                return -1
            solver2 = z3.Solver()
            solver2.add(base)
            if solver2.check(vb >= va) == z3.unsat:
                return 1
            s3 = z3.Solver()
            s3.add(base)
            if s3.check(va != vb) == z3.unsat:
                return 0
            return None

        return (entailed_sign(X[a], X[b]), entailed_sign(Y[a], Y[b]))


# ===================================================================== coref
def resolve_coreference(text: str) -> str:
    """Classical cross-sentence coreference (recency + compatibility over
    spaCy parses/NER — the standard rule-based method): pronoun subjects and
    demonstrative noun phrases are rewritten to their antecedents so the
    downstream readers see explicit subjects. Sentences that carry
    quantified/conditional structure keep their pronouns (those are rule
    variables, not references)."""
    from .syntax_tier import _nlp
    doc = _nlp()(text)
    out_sents = []
    salience: List[Tuple[str, str]] = []   # (surface, kind: person|thing)

    def antecedent(kind: str) -> Optional[str]:
        for surf, k in reversed(salience):
            if kind == "any" or k == kind:
                return surf
        return None

    for sent in doc.sents:
        stext = sent.text.strip()
        low = stext.lower()
        is_ruleish = bool(re.search(
            r"\b(if|then|when|every|all|each|someone|something|anyone)\b",
            low))
        replaced = stext
        if not is_ruleish:
            first = sent[0]
            # sentence-initial pronoun subject
            if first.lower_ in ("it", "he", "she", "they", "this", "these"):
                kind = ("person" if first.lower_ in ("he", "she")
                        else "any")
                ant = antecedent(kind)
                if ant:
                    if first.lower_ in ("this", "these") and len(sent) > 1:
                        # demonstrative NP: determiner + adjectives + ONE noun
                        np_end = None
                        for t in sent[1:]:
                            if t.pos_ == "ADJ":
                                continue
                            if t.pos_ in ("NOUN", "PROPN"):
                                np_end = t.i
                            break
                        if np_end is not None:
                            np = doc[first.i: np_end + 1].text
                            replaced = stext.replace(np, ant, 1)
                    else:
                        replaced = re.sub(rf"^{first.text}\b", ant, stext)
            # possessive pronouns mid-sentence: its/his/her/their
            for pron, kind in (("its", "any"), ("his", "person"),
                               ("her", "person"), ("their", "any")):
                if re.search(rf"\b{pron}\b", replaced.lower()):
                    ant = antecedent(kind)
                    if ant:
                        replaced = re.sub(rf"\b[Ii]ts\b|\b[Hh]is\b|"
                                          rf"\b[Hh]er\b|\b[Tt]heir\b",
                                          f"{ant}'s", replaced, count=1)
                        break
        out_sents.append(replaced)
        # update salience: if we resolved this sentence's subject, the
        # ANTECEDENT stays salient; otherwise record the actual subject
        if replaced != stext:
            continue
        for tok in sent:
            if tok.dep_ in ("nsubj", "nsubjpass") and tok.pos_ in \
                    ("PROPN", "NOUN") and tok.lower_ not in VAR_WORDS:
                span = doc[tok.left_edge.i: tok.i + 1].text
                span = re.sub(r"^(The|A|An|This|These|That|Those|Also,?)\s+",
                              "", span).strip()
                if not span:
                    continue
                kind = ("person" if tok.ent_type_ == "PERSON"
                        or (tok.pos_ == "PROPN"
                            and tok.ent_type_ in ("", "PERSON")) else "thing")
                if not salience or salience[-1][0] != span:
                    salience.append((span, kind))
    return " ".join(out_sents)


# ================================================================= quantities
def parse_quantity(sentence: str):
    """General measurement extraction: '<S> is 6,682 km long',
    '<S> goes around the Sun every 87.969 days', '<S> weighs 5 kg'.
    Returns (subject_raw, attribute, magnitude, unit) or None.
    The attribute comes from the adjective if present, else from the unit's
    physical dimension via pint ('km' -> length, 'days' -> time)."""
    import pint
    ureg = pint.UnitRegistry()
    tl = sentence.strip().rstrip(".")
    m = re.search(r"^(.*?)\b(?:is|are|was|were|measures?|weighs?|takes?|"
                  r"lasts?|spans?|stands?)\b(.*?)"
                  r"([\d][\d,]*\.?\d*)\s*([a-zA-Z]+)\b\s*([a-z]*)", tl)
    if not m:
        m = re.search(r"^(.*?)\b(?:every|once every|each)\b\s*"
                      r"([\d][\d,]*\.?\d*)\s*([a-zA-Z]+)\b", tl)
        if not m:
            return None
        subj, mag, unit = m.group(1), m.group(2), m.group(3)
        trailing = ""
        attr_hint = "period"
    else:
        subj, mid, mag, unit, trailing = (m.group(1), m.group(2),
                                          m.group(3), m.group(4), m.group(5))
        attr_hint = trailing.lower() if trailing else None
    try:
        q = float(mag.replace(",", "")) * ureg(unit.lower())
        base = q.to_base_units()
    except Exception:
        return None
    if attr_hint in ("long", "wide", "tall", "high", "deep"):
        attr = {"long": "length", "wide": "width", "tall": "height",
                "high": "height", "deep": "depth"}[attr_hint]
    elif attr_hint == "period":
        attr = "period"
    else:
        dim = str(base.dimensionality)
        attr = {"[length]": "length", "[time]": "duration",
                "[mass]": "mass", "[temperature]": "temperature"}.get(
                    dim, dim.strip("[]") or "quantity")
    # subject: the sentence's parsed nsubj span (robust to verb phrases)
    from .syntax_tier import _nlp
    doc = _nlp()(sentence)
    subj_tok = next((t for t in doc
                     if t.dep_ in ("nsubj", "nsubjpass")), None)
    if subj_tok is None:
        return None
    span = doc[subj_tok.left_edge.i: subj_tok.i + 1].text
    subj = norm_ent(span)
    if not subj or subj in VAR_WORDS:
        return None
    return (subj, attr, float(base.magnitude), str(base.units))
