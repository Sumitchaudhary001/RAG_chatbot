"""
Evaluation harness (Part 7).

Loads eval/test_questions.json (fill in 20 real questions/answers for
your ingested PDFs), runs each through the RAG chain in-process
(no server needed), and records:
    - retrieved documents (source + page)
    - generated answer
    - expected answer
    - an automatic correctness heuristic (word-overlap) as a first pass

The heuristic is intentionally crude -- it's a starting signal, not a
substitute for eyeballing the results. Review eval/eval_results.csv
and correct the "correct" column by hand before reporting final numbers.

Run with:  python -m eval.evaluate
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.chains.rag_chain import ask  # noqa: E402

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "test_questions.json")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "eval_results.csv")
RESULTS_JSON = os.path.join(os.path.dirname(__file__), "eval_results.json")


def word_overlap_score(expected: str, actual: str) -> float:
    """Crude heuristic: fraction of expected-answer words present in the
    generated answer. Not a substitute for manual review, but useful to
    flag obvious misses quickly."""
    expected_words = set(expected.lower().split())
    actual_words = set(actual.lower().split())
    if not expected_words:
        return 0.0
    return len(expected_words & actual_words) / len(expected_words)


def run_evaluation():
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    results = []
    for item in questions:
        question = item["question"].strip()
        expected = item["expected_answer"].strip()
        if not question:
            continue  # skip unfilled template rows

        session_id = f"eval-{item['id']}"
        answer, docs = ask(session_id, question)

        score = word_overlap_score(expected, answer)
        results.append(
            {
                "id": item["id"],
                "question": question,
                "expected_answer": expected,
                "generated_answer": answer,
                "retrieved_sources": [
                    f"{d.metadata.get('source')} p{d.metadata.get('page')}" for d in docs
                ],
                "overlap_score": round(score, 2),
                "correct": score > 0.5,  # heuristic default; review manually
            }
        )
        print(f"[{item['id']}] Q: {question}\n    A: {answer}\n    score={score:.2f}\n")

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "question", "expected_answer", "generated_answer",
                "retrieved_sources", "overlap_score", "correct",
            ],
        )
        writer.writeheader()
        for row in results:
            row = dict(row)
            row["retrieved_sources"] = "; ".join(row["retrieved_sources"])
            writer.writerow(row)

    n_correct = sum(1 for r in results if r["correct"])
    print(f"\n{n_correct}/{len(results)} answers passed the automatic heuristic.")
    print(f"Full results written to {RESULTS_CSV} and {RESULTS_JSON}")


if __name__ == "__main__":
    run_evaluation()
