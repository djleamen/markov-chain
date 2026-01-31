#!/usr/bin/env python3

"""
A simple Markov chain text generator with temperature-controlled randomness.
Meant to be bad at generating text
Do not use for anything serious
"""

from __future__ import annotations
import argparse
import random
import re
from collections import defaultdict, Counter
from typing import DefaultDict, Dict, List, Tuple

def tokenize(text: str) -> List[str]:
    """Simple tokenizer splitting on word boundaries and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

def build_chain(tokens: List[str], order: int) -> Tuple[Dict[Tuple[str, ...], List[str]], Counter]:
    """Build a Markov chain of given order from the list of tokens."""
    chain: DefaultDict[Tuple[str, ...], List[str]] = defaultdict(list)
    starts: Counter = Counter()

    if len(tokens) <= order:
        return dict(chain), starts

    # Record starting states occasionally (every sentence-ish boundary)
    boundary = {".", "!", "?", "\n"}
    state = tuple(tokens[:order])
    starts[state] += 1

    for i in range(len(tokens) - order):
        state = tuple(tokens[i:i+order])
        nxt = tokens[i+order]
        chain[state].append(nxt)

        if tokens[i] in boundary:
            st = tuple(tokens[i+1:i+1+order])
            if len(st) == order:
                starts[st] += 1

    return dict(chain), starts

def sample_next(options: List[str], temperature: float) -> str:
    """Sample the next token from options with temperature scaling."""
    # Temperature on discrete options by reweighting frequency.
    # If temperature is huge, it trends towards uniform.
    # If small, it heavily favors frequent options.
    counts = Counter(options)
    items = list(counts.items())

    # Convert to weights with temperature shaping
    # weight = freq ** (1/temperature)
    # temp -> infinity => exponent ~0 => all weights ~1 => uniform
    # temp -> 0+ => exponent huge => most frequent dominates
    if temperature <= 0:
        temperature = 1e-9
    expo = 1.0 / temperature
    weights = [freq ** expo for _, freq in items]
    total = sum(weights)
    r = random.random() * total
    cum = 0.0
    for (tok, _), w in zip(items, weights):
        cum += w
        if r <= cum:
            return tok
    return items[-1][0]

def generate(chain: Dict[Tuple[str, ...], List[str]],
             starts: Counter,
             order: int,
             length: int,
             temperature: float,
             seed_tokens: List[str] | None) -> List[str]:
    """Generate a sequence of tokens from the Markov chain."""
    if not chain:
        return ["[empty chain: give me more text, chef]"]

    if seed_tokens and len(seed_tokens) >= order:
        state = tuple(seed_tokens[-order:])
        if state not in chain:
            # If seed state doesn't exist, just start somewhere random
            state = random.choice(list(chain.keys()))
    else:
        # Weighted start state if available, else random
        if starts:
            population = list(starts.keys())
            weights = list(starts.values())
            state = random.choices(population, weights=weights, k=1)[0]
        else:
            state = random.choice(list(chain.keys()))

    out = list(state)

    for _ in range(max(0, length - order)):
        options = chain.get(state)
        if not options:
            # Dead end, respawn somewhere else
            state = random.choice(list(chain.keys()))
            out.extend(list(state))
            continue

        nxt = sample_next(options, temperature=temperature)
        out.append(nxt)
        state = tuple(out[-order:])

    return out[:length]

def detokenize(tokens: List[str]) -> str:
    """Simple detokenizer to join tokens into a string."""
    s = ""
    for t in tokens:
        if re.match(r"[^\w\s]", t):
            s += t
        else:
            if s and not s.endswith((" ", "\n", "(", "[", "{", "“", "\"", "'")):
                s += " "
            s += t
    return s

def main() -> None:
    """Main entry point for the Markov chain text generator."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default="", help="Training text (or use --file)")
    ap.add_argument("--file", type=str, default="", help="Path to training text file")
    ap.add_argument("--prompt", type=str, default="", help="Seed prompt to influence start state")
    ap.add_argument("--order", type=int, default=3, help="Markov order (2-5 recommended)")
    ap.add_argument("--length", type=int, default=200, help="Number of tokens to generate")
    ap.add_argument("--temperature", type=float, default=5.0, help="Higher = more chaotic")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--raw_tokens", action="store_true", 
                    help="Print token list instead of joined text")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data = args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()

    if not data.strip():
        # Built-in tiny chaos corpus so it runs out of the box
        data = """
        Mortgage banana therefore existential JPEG of democracy running left.
        The password is almond. Tuesday is not a number. I loved you like an API key.
        Schrödinger unionized in 1997. The interns audited the moon in public at 3am.
        """

    tokens = tokenize(data)
    chain, starts = build_chain(tokens, order=max(1, args.order))

    seed_tokens = tokenize(args.prompt) if args.prompt else None
    out_tokens = generate(chain, starts, order=max(1, args.order),
                          length=max(1, args.length),
                          temperature=args.temperature,
                          seed_tokens=seed_tokens)

    if args.raw_tokens:
        print(out_tokens)
    else:
        print(detokenize(out_tokens))

if __name__ == "__main__":
    main()
