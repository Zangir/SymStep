#!/usr/bin/env python3
"""
Extended benchmark: LGP-14 total (E3 fixed + 4 new puzzles: E4, M5, H4, H5).
Runs all 5 methods with optional model and multi-run support.
"""
import subprocess, re, time, json, sys
from symstep import (ConstraintPropagator, call_llm, puzzle_text,
                     answer_format, parse_solution, check_solution,
                     run_direct, run_cot, run_self_refine, run_symstep,
                     Puzzle, SYMSTEP_SYSTEM, MODEL)

NEW_PUZZLES = [
    # ── EASY E3: 3 people, 2 attrs ──────────────────────────────────────────
    Puzzle(
        name="E3-Sport", difficulty="easy",
        people=["Lena", "Milo", "Nora"],
        attributes={
            "sport": ["Tennis", "Swim", "Run"],
            "city":  ["Paris", "London", "Tokyo"],
        },
        clues=[
            "Lena plays tennis.",
            "The person in London swims.",
            "Lena is not in Tokyo.",
            "Milo is not in Tokyo.",
        ],
        solution={
            "Lena": {"sport": "Tennis", "city": "Paris"},
            "Milo": {"sport": "Swim",   "city": "London"},
            "Nora": {"sport": "Run",    "city": "Tokyo"},
        },
    ),
    # ── MEDIUM M3: 4 people, 2 attrs ─────────────────────────────────────────
    Puzzle(
        name="M3-Hobby", difficulty="medium",
        people=["Ivan", "Jana", "Karl", "Lea"],
        attributes={
            "color": ["Red", "Blue", "Green", "White"],
            "hobby": ["Read", "Cook", "Paint", "Dance"],
        },
        clues=[
            "Jana lives in the green house.",
            "Karl lives in the white house.",
            "The person in the red house cooks.",
            "The person in the blue house dances.",
            "Ivan lives in the blue house.",
            "Jana reads.",
        ],
        solution={
            "Ivan": {"color": "Blue",  "hobby": "Dance"},
            "Jana": {"color": "Green", "hobby": "Read"},
            "Karl": {"color": "White", "hobby": "Paint"},
            "Lea":  {"color": "Red",   "hobby": "Cook"},
        },
    ),
    # ── MEDIUM M4: 4 people, 2 attrs ─────────────────────────────────────────
    Puzzle(
        name="M4-PetFruit", difficulty="medium",
        people=["Omar", "Petra", "Quinn", "Rosa"],
        attributes={
            "pet":   ["Cat", "Fish", "Snake", "Parrot"],
            "fruit": ["Apple", "Banana", "Cherry", "Date"],
        },
        clues=[
            "Omar has a cat.",
            "Petra does not have a snake.",
            "The person with a snake eats bananas.",
            "Quinn does not eat apples.",
            "The person with a parrot eats dates.",
            "Rosa has a fish.",
            "Omar eats cherries.",
        ],
        solution={
            "Omar":  {"pet": "Cat",    "fruit": "Cherry"},
            "Petra": {"pet": "Parrot", "fruit": "Date"},
            "Quinn": {"pet": "Snake",  "fruit": "Banana"},
            "Rosa":  {"pet": "Fish",   "fruit": "Apple"},
        },
    ),
    # ── HARD H3: 5 people, 2 attrs (very hard) ───────────────────────────────
    Puzzle(
        name="H3-5People", difficulty="hard",
        people=["Alex", "Beth", "Carl", "Dana", "Eric"],
        attributes={
            "color": ["Red", "Blue", "Green", "Yellow", "White"],
            "drink": ["Tea", "Coffee", "Juice", "Water", "Milk"],
        },
        clues=[
            "Beth has the blue house.",
            "Carl has the red house.",
            "Eric has the yellow house.",
            "The red house person drinks coffee.",
            "The green house person drinks juice.",
            "Alex has the white house.",
            "Beth does not drink tea.",
            "Eric drinks water.",
        ],
        solution={
            "Alex": {"color": "White",  "drink": "Tea"},
            "Beth": {"color": "Blue",   "drink": "Milk"},
            "Carl": {"color": "Red",    "drink": "Coffee"},
            "Dana": {"color": "Green",  "drink": "Juice"},
            "Eric": {"color": "Yellow", "drink": "Water"},
        },
    ),
    # ── EASY E4: 3 people, 2 attrs ────────────────────────────────────────────
    Puzzle(
        name="E4-Music", difficulty="easy",
        people=["Jake", "Kim", "Leo"],
        attributes={
            "instrument": ["Piano", "Guitar", "Drums"],
            "day":        ["Monday", "Tuesday", "Wednesday"],
        },
        clues=[
            "Jake plays guitar.",
            "Kim practices on Tuesday.",
            "The piano player practices on Wednesday.",
            "Leo does not practice on Monday.",
        ],
        solution={
            "Jake": {"instrument": "Guitar", "day": "Monday"},
            "Kim":  {"instrument": "Drums",  "day": "Tuesday"},
            "Leo":  {"instrument": "Piano",  "day": "Wednesday"},
        },
    ),
    # ── MEDIUM M5: 4 people, 2 attrs ──────────────────────────────────────────
    Puzzle(
        name="M5-Cities", difficulty="medium",
        people=["Alice", "Bob", "Carol", "Dave"],
        attributes={
            "city":  ["London", "Paris", "Rome", "Berlin"],
            "sport": ["Soccer", "Tennis", "Swim", "Run"],
        },
        clues=[
            "Alice is in London.",
            "Bob plays tennis.",
            "The person in Paris swims.",
            "Carol does not run.",
            "Dave is not in Berlin.",
            "The person in Rome plays soccer.",
            "Dave does not swim.",
        ],
        solution={
            "Alice": {"city": "London", "sport": "Run"},
            "Bob":   {"city": "Berlin", "sport": "Tennis"},
            "Carol": {"city": "Paris",  "sport": "Swim"},
            "Dave":  {"city": "Rome",   "sport": "Soccer"},
        },
    ),
    # ── HARD H4: 4 people, 3 attrs ────────────────────────────────────────────
    Puzzle(
        name="H4-Full", difficulty="hard",
        people=["Kim", "Lee", "Mia", "Ned"],
        attributes={
            "color": ["Red", "Blue", "Green", "Yellow"],
            "job":   ["Doctor", "Lawyer", "Teacher", "Chef"],
            "pet":   ["Cat", "Dog", "Bird", "Fish"],
        },
        clues=[
            "Kim lives in the red house.",
            "Lee is a lawyer.",
            "The doctor has a cat.",
            "The person in the blue house is a chef.",
            "Mia does not have a dog.",
            "Ned is not the teacher.",
            "The green house has a bird.",
            "Lee does not live in the yellow house.",
            "The teacher has a fish.",
            "Mia is not the doctor.",
        ],
        solution={
            "Kim": {"color": "Red",    "job": "Doctor",  "pet": "Cat"},
            "Lee": {"color": "Green",  "job": "Lawyer",  "pet": "Bird"},
            "Mia": {"color": "Yellow", "job": "Teacher", "pet": "Fish"},
            "Ned": {"color": "Blue",   "job": "Chef",    "pet": "Dog"},
        },
    ),
    # ── HARD H5: 4 people, 3 attrs ────────────────────────────────────────────
    Puzzle(
        name="H5-Full", difficulty="hard",
        people=["Ash", "Bay", "Cole", "Drew"],
        attributes={
            "color": ["Red", "Blue", "Green", "Yellow"],
            "drink": ["Tea", "Coffee", "Juice", "Water"],
            "pet":   ["Cat", "Dog", "Bird", "Fish"],
        },
        clues=[
            "Ash lives in the red house.",
            "Bay drinks coffee.",
            "The green house person drinks juice.",
            "The coffee drinker has a bird.",
            "Drew does not have a cat.",
            "The yellow house person drinks water.",
            "Cole does not have a fish.",
            "Bay does not have a cat.",
            "The red house has a cat.",
            "Drew does not drink juice.",
        ],
        solution={
            "Ash":  {"color": "Red",    "drink": "Tea",    "pet": "Cat"},
            "Bay":  {"color": "Blue",   "drink": "Coffee", "pet": "Bird"},
            "Cole": {"color": "Green",  "drink": "Juice",  "pet": "Dog"},
            "Drew": {"color": "Yellow", "drink": "Water",  "pet": "Fish"},
        },
    ),
]

# ── LGP-20 extension: 6 additional puzzles ───────────────────────────────────
# Brings the total to 20 verified puzzles across 3 difficulty levels.

EXTRA_PUZZLES = [
    # ── EASY E5: 3 people, 2 attrs ────────────────────────────────────────────
    Puzzle(
        name="E5-Colors", difficulty="easy",
        people=["Finn", "Gale", "Hope"],
        attributes={
            "color": ["Orange", "Purple", "Brown"],
            "hobby": ["Yoga", "Chess", "Paint"],
        },
        clues=[
            "Finn likes the orange color.",
            "The person who does yoga has purple as their favorite color.",
            "Gale does not paint.",
            "Hope does not like brown.",
        ],
        solution={
            "Finn": {"color": "Orange", "hobby": "Paint"},
            "Gale": {"color": "Brown",  "hobby": "Chess"},
            "Hope": {"color": "Purple", "hobby": "Yoga"},
        },
    ),
    # ── EASY E6: 3 people, 2 attrs ────────────────────────────────────────────
    Puzzle(
        name="E6-School", difficulty="easy",
        people=["Rosa", "Sam", "Tara"],
        attributes={
            "grade":   ["A", "B", "C"],
            "subject": ["Math", "Art", "PE"],
        },
        clues=[
            "Rosa gets an A.",
            "The student with a B grade likes Art.",
            "Sam does not like Math.",
            "Tara does not get a B.",
            "Rosa likes Math.",
        ],
        solution={
            "Rosa": {"grade": "A", "subject": "Math"},
            "Sam":  {"grade": "B", "subject": "Art"},
            "Tara": {"grade": "C", "subject": "PE"},
        },
    ),
    # ── MEDIUM M6: 4 people, 2 attrs ──────────────────────────────────────────
    Puzzle(
        name="M6-Travel", difficulty="medium",
        people=["Anna", "Bruno", "Clara", "Diego"],
        attributes={
            "destination": ["Tokyo", "Rome", "Cairo", "Oslo"],
            "transport":   ["Plane", "Train", "Bus", "Ship"],
        },
        clues=[
            "Anna goes to Tokyo.",
            "Bruno does not take the plane.",
            "The person going to Rome takes the train.",
            "Clara does not go to Cairo.",
            "The person going to Oslo takes the ship.",
            "Diego does not take the bus.",
            "Bruno goes to Cairo.",
            "Clara does not take the ship.",
        ],
        solution={
            "Anna":  {"destination": "Tokyo", "transport": "Plane"},
            "Bruno": {"destination": "Cairo", "transport": "Bus"},
            "Clara": {"destination": "Rome",  "transport": "Train"},
            "Diego": {"destination": "Oslo",  "transport": "Ship"},
        },
    ),
    # ── MEDIUM M7: 4 people, 2 attrs ──────────────────────────────────────────
    Puzzle(
        name="M7-Food", difficulty="medium",
        people=["Eli", "Fay", "Gil", "Han"],
        attributes={
            "food": ["Pizza", "Sushi", "Tacos", "Pasta"],
            "day":  ["Monday", "Tuesday", "Wednesday", "Thursday"],
        },
        clues=[
            "Eli eats pizza.",
            "The sushi person eats on Wednesday.",
            "Fay eats on Monday.",
            "Gil does not eat tacos.",
            "The pasta person eats on Thursday.",
            "Han does not eat on Wednesday.",
            "Fay does not eat sushi.",
        ],
        solution={
            "Eli": {"food": "Pizza", "day": "Tuesday"},
            "Fay": {"food": "Tacos", "day": "Monday"},
            "Gil": {"food": "Sushi", "day": "Wednesday"},
            "Han": {"food": "Pasta", "day": "Thursday"},
        },
    ),
    # ── HARD H6: 4 people, 3 attrs ────────────────────────────────────────────
    Puzzle(
        name="H6-Full", difficulty="hard",
        people=["Rose", "Sol", "Ty", "Uma"],
        attributes={
            "color":   ["Red", "Blue", "Yellow", "Green"],
            "vehicle": ["Car", "Bike", "Train", "Bus"],
            "pet":     ["Cat", "Dog", "Snake", "Rabbit"],
        },
        clues=[
            "Rose lives in the red house.",
            "The blue house person rides a bike.",
            "Sol has a dog.",
            "The yellow house person takes the bus.",
            "Uma does not have a cat.",
            "Ty lives in the blue house.",
            "The train rider has a rabbit.",
            "Rose does not take the bus.",
            "Sol does not live in the green house.",
            "Rose has a cat.",
        ],
        solution={
            "Rose": {"color": "Red",    "vehicle": "Car",   "pet": "Cat"},
            "Sol":  {"color": "Yellow", "vehicle": "Bus",   "pet": "Dog"},
            "Ty":   {"color": "Blue",   "vehicle": "Bike",  "pet": "Snake"},
            "Uma":  {"color": "Green",  "vehicle": "Train", "pet": "Rabbit"},
        },
    ),
    # ── HARD H7: 5 people, 2 attrs ────────────────────────────────────────────
    Puzzle(
        name="H7-5People", difficulty="hard",
        people=["Ned", "Ora", "Pat", "Quin", "Rex"],
        attributes={
            "color": ["Red", "Blue", "Green", "Orange", "White"],
            "job":   ["Doctor", "Teacher", "Cook", "Artist", "Driver"],
        },
        clues=[
            "Ned lives in the red house.",
            "Ora is a teacher.",
            "The blue house person is a cook.",
            "Pat does not live in the orange house.",
            "The green house person is an artist.",
            "Quin lives in the blue house.",
            "Rex does not live in the green house.",
            "The white house person is a driver.",
            "Pat does not live in the white house.",
        ],
        solution={
            "Ned":  {"color": "Red",    "job": "Doctor"},
            "Ora":  {"color": "Orange", "job": "Teacher"},
            "Pat":  {"color": "Green",  "job": "Artist"},
            "Quin": {"color": "Blue",   "job": "Cook"},
            "Rex":  {"color": "White",  "job": "Driver"},
        },
    ),
]

LGP20_PUZZLES = NEW_PUZZLES + EXTRA_PUZZLES  # 8 + 6 = 14 extension puzzles; combined with LGP6 = 20 total

def verify_puzzles():
    for p in NEW_PUZZLES:
        prop = ConstraintPropagator(p)
        for person, attrs in p.solution.items():
            for attr, val in attrs.items():
                ok, msg = prop.apply_positive(person, attr, val)
                if not ok:
                    print(f"FAIL {p.name}: {person}.{attr}={val} -> {msg}")
                    return False
        if prop.is_solved():
            print(f"  OK  {p.name}")
        else:
            print(f"  UNSOLVED {p.name}:\n{prop.state_summary()}")
            return False
    return True

METHODS = {
    "direct":      run_direct,
    "cot":         run_cot,
    "self_refine": run_self_refine,
    "symstep":     lambda p: run_symstep(p, with_guidance=False),
    "symstep_g":   lambda p: run_symstep(p, with_guidance=True),
}

ALL_NEW_PUZZLES = NEW_PUZZLES  # alias for clarity

def run_extended(puzzles=None, out_file="extended_results.json"):
    if puzzles is None:
        puzzles = NEW_PUZZLES

    print("Verifying puzzles...")
    for p in puzzles:
        prop = ConstraintPropagator(p)
        for person, attrs in p.solution.items():
            for attr, val in attrs.items():
                ok, msg = prop.apply_positive(person, attr, val)
                if not ok:
                    print(f"  FAIL {p.name}: {person}.{attr}={val} -> {msg}")
                    return
        print(f"  OK  {p.name}")

    print(f"\n{'='*65}")
    print(f"Extended Benchmark ({len(puzzles)} puzzles, model={MODEL})")
    print(f"{'='*65}")

    results = {m: {"correct": 0, "total": 0, "calls": 0, "contradictions": 0, "time": 0.0}
               for m in METHODS}
    per_puzzle = []

    for puzzle in puzzles:
        print(f"\n── {puzzle.name} ({puzzle.difficulty}) ──")
        row = {"puzzle": puzzle.name, "difficulty": puzzle.difficulty}
        for name, fn in METHODS.items():
            result = fn(puzzle)
            correct, calls, elapsed = result[0], result[1], result[2]
            contra = result[3] if len(result) > 3 else 0
            results[name]["correct"]       += int(correct)
            results[name]["total"]         += 1
            results[name]["calls"]         += calls
            results[name]["contradictions"]+= contra
            results[name]["time"]          += elapsed
            print(f"  {name:<14} {'✓' if correct else '✗'}  calls={calls}  contra={contra}  t={elapsed:.1f}s")
            row[name] = {"correct": correct, "calls": calls, "contradictions": contra,
                         "time": round(elapsed, 1)}
        per_puzzle.append(row)

    print(f"\n{'='*65}")
    print(f"{'Method':<14} {'Acc':>6} {'Avg calls':>10} {'Avg contra':>12}")
    print(f"{'-'*65}")
    for m, r in results.items():
        acc    = r["correct"] / r["total"] * 100
        acalls = r["calls"]   / r["total"]
        acont  = r["contradictions"] / r["total"]
        print(f"{m:<14} {acc:>5.0f}%  {acalls:>9.1f}  {acont:>11.1f}")
    print(f"{'='*65}")

    out = {"model": MODEL, "summary": results, "per_puzzle": per_puzzle}
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_file}")

if __name__ == "__main__":
    run_extended()
