"""Run the automatic steps for one topic: fetch -> score (all three) -> extract -> graph.

    python run_all.py --topic "LLM evaluation and reliability for clinical and health text"

Labeling and evaluation need a human, so run them yourself afterward:
    python label.py --topic "..." --k 150
    python evaluate.py --topic "..."
"""
import argparse

import config
import extract
import fetch
import graph
import score


def main(topic, n, criteria, top):
    fetch.main(topic, n)
    score.main(topic, ["minilm", "bge", "llm"], criteria)
    extract.main(topic, top)
    graph.main(topic)
    print("\nNext: label papers, then compare scorers:")
    print(f'  python label.py --topic "{topic}" --k 150')
    print(f'  python evaluate.py --topic "{topic}"')


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--topic", default=config.DEMO_TOPIC)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--criteria", default=config.DEFAULT_CRITERIA)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    main(args.topic, args.n, args.criteria, args.top)
