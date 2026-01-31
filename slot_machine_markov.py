#!/usr/bin/env python3

"""
Character-level Markov chain
Even worse than word-level
Meant to be bad at generating text
So bad that I've called it a slot machine
Do not use for anything serious
"""

from __future__ import annotations
import argparse
import random
from collections import defaultdict, Counter
from typing import DefaultDict, Dict, List, Tuple

def build_char_chain(text: str, order: int) -> Tuple[Dict[str, List[str]], Counter]:
    """Build a character-level Markov chain of given order from the text."""
    chain: DefaultDict[str, List[str]] = defaultdict(list)
    starts: Counter = Counter()

    if len(text) <= order:
        return dict(chain), starts

    # starts are every position after a newline
    # starting mid-word is fun but unreadable
    for i in range(len(text) - order):
        state = text[i:i+order]
        nxt = text[i+order]
        chain[state].append(nxt)
        if i == 0 or text[i-1] == "\n":
            starts[state] += 1

    return dict(chain), starts

def sample_next(options: List[str], temperature: float) -> str:
    """Sample the next character from options with temperature scaling."""
    counts = Counter(options)
    items = list(counts.items())
    if temperature <= 0:
        temperature = 1e-9
    expo = 1.0 / temperature
    weights = [freq ** expo for _, freq in items]
    total = sum(weights)
    r = random.random() * total
    cum = 0.0
    for (ch, _), w in zip(items, weights):
        cum += w
        if r <= cum:
            return ch
    return items[-1][0]

def generate(chain: Dict[str, List[str]], starts: Counter, 
             order: int, length: int, temperature: float, prompt: str) -> str:
    """Generate a sequence of characters from the Markov chain."""
    if not chain:
        return ""

    if prompt and len(prompt) >= order:
        state = prompt[-order:]
        if state not in chain:
            state = random.choice(list(chain.keys()))
    else:
        if starts:
            population = list(starts.keys())
            weights = list(starts.values())
            state = random.choices(population, weights=weights, k=1)[0]
        else:
            state = random.choice(list(chain.keys()))

    out = state
    while len(out) < length:
        options = chain.get(state)
        if not options:
            state = random.choice(list(chain.keys()))
            out += state
            continue
        nxt = sample_next(options, temperature=temperature)
        out += nxt
        state = out[-order:]
    return out[:length]

def main() -> None:
    """Main entry point for the character-level Markov chain text generator."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default="")
    ap.add_argument("--file", type=str, default="")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--length", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data = args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()
    if not data.strip():
        data = "Mortgage banana therefore existential JPEG of democracy running left.\nThe password is almond.\nTuesday.\n"

    chain, starts = build_char_chain(data, order=max(1, args.order))
    print(generate(chain, starts, order=max(1, args.order), 
                   length=max(1, args.length), temperature=args.temperature, 
                   prompt=args.prompt))

if __name__ == "__main__":
    main()
