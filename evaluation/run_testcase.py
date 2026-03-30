#!/usr/bin/env python3
"""
Load test cases from tests/testcases.json and run them against the CE-GPT API.
Creates a session, then POSTs each case to /api/v1/generate/stream, parses SSE,
and logs start_generate, stop_generate, content (concatenated chunks), and sources.

Usage:
  python tests/run_testcases.py                    # list all cases
  python tests/run_testcases.py --category "Curriculum & Credit Structure"
  python tests/run_testcases.py --id curriculum-total-credits
  python tests/run_testcases.py --run --url http://10.240.68.50:8000
  python tests/run_testcases.py --run --out results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import string
import sys
import time
def load_testcases(path: str | None = None) -> dict:
    path = path or os.path.join(os.path.dirname(__file__), "dataset/testcase_1.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def random_user_id(prefix: str = "run-", length: int = 12) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def create_session(base_url: str, user_id: str, requests_mod) -> str:
    r = requests_mod.post(
        f"{base_url}/api/v1/sessions",
        json={"user_id": user_id, "metadata": {}, "ttl_hours": 1},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    return data["session_id"]


def run_stream(base_url: str, session_id: str, user_id: str, query: str, language: str, requests_mod):
    """POST to generate/stream, parse SSE. Returns (query_sent_at, start_generate, stop_generate, full_content, sources)."""
    start_generate = None
    stop_generate = None
    full_content_parts = []
    sources_log = None

    payload = {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        "language": language if language != "mixed" else "auto",
        "include_sources": True,
    }

    query_sent_at = time.time()
    with requests_mod.post(
        f"{base_url}/api/v1/generate/stream",
        json=payload,
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:].strip())
            except json.JSONDecodeError:
                continue
            msg_type = data.get("type")
            if msg_type == "chunk":
                content = data.get("content", "")
                if isinstance(content, str):
                    # Already decoded by json.loads (e.g. \u0e04 -> actual char)
                    pass
                else:
                    content = str(content)
                if start_generate is None:
                    start_generate = time.time()
                full_content_parts.append(content)
                stop_generate = time.time()
                # Log this chunk in readable form (UTF-8)
                print(f"  [content] {content!r}", flush=True)
            elif msg_type == "sources":
                sources_log = data.get("sources", [])
                print(f"  [sources] {len(sources_log)} source(s)", flush=True)
                for i, src in enumerate(sources_log):
                    preview = (src.get("content") or "")[:80].replace("\n", " ")
                    print(f"    source[{i}]: {preview!r}...", flush=True)

    full_content = "".join(full_content_parts)
    return query_sent_at, start_generate, stop_generate, full_content, sources_log


def main() -> None:
    parser = argparse.ArgumentParser(description="CE-GPT test case runner")
    parser.add_argument("--path", default=None, help="Path to testcases.json")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--id", type=str, help="Single case id")
    parser.add_argument("--run", action="store_true", help="Run queries against API (session + stream)")
    parser.add_argument(
        "--url",
        type=str,
        default="http://10.240.68.50:8000",
        help="API base URL when --run",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="user_id for session (default: random). Used with same value for generate/stream.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (e.g. results.csv). Columns: case_id, query, duration_before_stream_sec, duration_until_finish_sec, stream_duration_sec, content, sources.",
    )
    args = parser.parse_args()

    data = load_testcases(args.path)
    cases = data["cases"]

    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
        if not cases:
            print(f"No case with id: {args.id}", file=sys.stderr)
            sys.exit(1)
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]

    if not args.run:
        for case in cases:
            print(f"{case['id']}\t[{case['category']}]\t{case['query'][:70]}...")
        return

    try:
        import requests
    except ImportError:
        print("Install requests to use --run: pip install requests", file=sys.stderr)
        sys.exit(1)

    base = args.url.rstrip("/")
    user_id = args.user_id or random_user_id()
    print(f"Creating session for user_id={user_id!r}...", flush=True)
    try:
        session_id = create_session(base, user_id, requests)
        print(f"session_id={session_id!r}", flush=True)
    except Exception as e:
        print(f"Failed to create session: {e}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for case in cases:
        q = case.get("query", "").strip()
        if not q or q.startswith("[Do not use"):
            print(f"[SKIP] {case['id']}: {q[:60]}...")
            continue
        case_id = case["id"]
        language = case.get("language", "auto")
        print(f"\n--- {case_id} ---", flush=True)
        try:
            query_sent_at, start_generate, stop_generate, content, sources = run_stream(
                base, session_id, user_id, q, language, requests
            )
            # Durations in seconds: how long before model starts streaming, how long until it finishes
            duration_before_stream = (start_generate - query_sent_at) if start_generate is not None else None
            duration_until_finish = (stop_generate - query_sent_at) if stop_generate is not None else None
            stream_duration = (stop_generate - start_generate) if (start_generate and stop_generate) else None
            if duration_before_stream is not None:
                print(f"  duration_before_stream_sec={duration_before_stream:.3f}", flush=True)
            if duration_until_finish is not None:
                print(f"  duration_until_finish_sec={duration_until_finish:.3f}", flush=True)
            if stream_duration is not None:
                print(f"  stream_duration_sec={stream_duration:.3f}", flush=True)
            rows.append(
                {
                    "case_id": case_id,
                    "query": q,
                    "duration_before_stream_sec": f"{duration_before_stream:.3f}" if duration_before_stream is not None else "",
                    "duration_until_finish_sec": f"{duration_until_finish:.3f}" if duration_until_finish is not None else "",
                    "stream_duration_sec": f"{stream_duration:.3f}" if stream_duration is not None else "",
                    "content": content,
                    "sources": json.dumps(sources, ensure_ascii=False) if sources is not None else "",
                }
            )
            print(f"[OK]  {case_id}", flush=True)
        except Exception as e:
            print(f"[ERR] {case_id}: {e}", flush=True)
            rows.append(
                {
                    "case_id": case_id,
                    "query": q,
                    "duration_before_stream_sec": "",
                    "duration_until_finish_sec": "",
                    "stream_duration_sec": "",
                    "content": "",
                    "sources": json.dumps({"error": str(e)}, ensure_ascii=False),
                }
            )

    if args.out and rows:
        fieldnames = [
            "case_id",
            "query",
            "duration_before_stream_sec",
            "duration_until_finish_sec",
            "stream_duration_sec",
            "content",
            "sources",
        ]
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} row(s) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
