import asyncio
import aiohttp
import time
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator


async def create_session(session, base_url, user_id):
    async with session.post(
        f"{base_url}/api/v1/sessions",
        json={"user_id": user_id, "metadata": {}, "ttl_hours": 1}
    ) as r:
        r.raise_for_status()
        data = await r.json()
        return data["session_id"]

async def run_single(session, base_url, queries):
    user_id = f"test-user-{time.time_ns()}"
    import random
    query = random.choice(queries) if isinstance(queries, list) else queries
    try:
        session_id = await create_session(session, base_url, user_id)
        payload = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "language": "auto",
            "include_sources": True,
        }
        start = time.time()
        ttft = None
        async with session.post(f"{base_url}/api/v1/generate/stream", json=payload) as r:
            r.raise_for_status()
            async for line in r.content:
                if ttft is None:
                    ttft = time.time() - start
        end = time.time()
        return {"latency": end - start, "ttft": ttft, "success": True}
    except Exception as e:
        print(f"Error: {e}")
        return {"latency": 0, "ttft": 0, "success": False}

async def run_concurrency_level(base_url, concurrency_level, queries):
    print(f"Testing concurrency: {concurrency_level}")
    timeout = aiohttp.ClientTimeout(total=300)
    start_time = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [run_single(session, base_url, queries) for _ in range(concurrency_level)]
        results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    successes = [r for r in results if r["success"]]
    if not successes:
        return 0, 0, 0, 0
    
    avg_latency = sum(r["latency"] for r in successes) / len(successes)
    avg_ttft = sum(r["ttft"] for r in successes) / len(successes)
    success_rate = len(successes) / concurrency_level
    
    total_time = end_time - start_time
    throughput = len(successes) / total_time
    
    print(f"  Avg Latency: {avg_latency:.2f}s, Avg TTFT: {avg_ttft:.2f}s, Success: {success_rate*100:.0f}%, Throughput: {throughput:.2f} req/s")
    return avg_latency, avg_ttft, success_rate, throughput

def find_elbow(x, y):
    if KneeLocator:
        kneedle = KneeLocator(x, y, S=1.0, curve="convex", direction="increasing")
        return kneedle.elbow
    else:
        # Fallback simple heuristic: max distance from line segment connecting first and last points
        x = np.array(x)
        y = np.array(y)
        
        # Line between first and last point
        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])
        
        distances = []
        for i in range(len(x)):
            p3 = np.array([x[i], y[i]])
            distance = np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)
            distances.append(distance)
            
        elbow_idx = np.argmax(distances)
        return x[elbow_idx]

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://10.240.68.50:8000")
    parser.add_argument("--query", default="List 5 computer engineering courses.")
    parser.add_argument("--testcases", default="tests/dataset/testcase_1.json", help="Path to testcases JSON file to randomly select queries from (overrides --query)")
    parser.add_argument("--levels", default="1,3,5,10,15,20", help="Comma separated concurrency levels")
    parser.add_argument("--out", default="performance_elbow.png", help="Output plot image name")
    parser.add_argument("--csv", default="performance_results.csv", help="Output CSV results")
    args = parser.parse_args()
    
    queries = args.query
    if args.testcases:
        try:
            with open(args.testcases, 'r', encoding='utf-8') as f:
                data = json.load(f)
                queries = [c["query"] for c in data.get("cases", []) if "query" in c and not c["query"].startswith("[Do not use")]
            if not queries:
                print(f"No valid queries found in {args.testcases}, falling back to --query")
                queries = args.query
        except Exception as e:
            print(f"Failed to load testcases from {args.testcases}: {e}")
    
    levels = [int(i) for i in args.levels.split(",")]
    avg_latencies = []
    avg_ttfts = []
    success_rates = []
    throughputs = []
    
    print(f"Starting concurrency performance test against {args.url}")
    if isinstance(queries, list):
        print(f"Query Source: Randomly selecting from {len(queries)} cases in {args.testcases}")
    else:
        print(f"Query: {args.query}")
    print(f"Levels: {levels}")
    print("-" * 40)
    
    for level in levels:
        lat, ttft, sr, thr = await run_concurrency_level(args.url, level, queries)
        avg_latencies.append(lat)
        avg_ttfts.append(ttft)
        success_rates.append(sr)
        throughputs.append(thr)
        
    elbow = find_elbow(levels, avg_latencies)
    print("-" * 40)
    print(f"Identified elbow point at concurrency: {elbow}")
    
    # Write to CSV
    import csv
    with open(args.csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Concurrency Level', 'Avg Latency (s)', 'Avg TTFT (s)', 'Success Rate', 'Throughput (req/s)'])
        for i in range(len(levels)):
            writer.writerow([levels[i], avg_latencies[i], avg_ttfts[i], success_rates[i], throughputs[i]])
    print(f"Results saved to {args.csv}")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("API Concurrency Performance Tests", fontsize=16)

    # 1. Latency Plot
    ax = axes[0, 0]
    ax.plot(levels, avg_latencies, marker='o', linestyle='-', linewidth=2, label='Avg Latency (s)')
    if elbow:
        elbow_index = levels.index(elbow)
        ax.plot(elbow, avg_latencies[elbow_index], 'ro', markersize=10, label=f'Elbow ({elbow})')
        ax.axvline(x=elbow, color='r', linestyle='--', alpha=0.5)
        
    ax.set_title("Concurrency vs Latency")
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Average Latency (seconds)")
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend()
    
    # 2. Throughput Plot
    ax2 = axes[0, 1]
    ax2.plot(levels, throughputs, marker='s', linestyle='-', linewidth=2, color='green', label='Throughput')
    ax2.set_title("Concurrency vs Throughput")
    ax2.set_xlabel("Concurrent Requests")
    ax2.set_ylabel("Throughput (req/sec)")
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    
    # 3. TTFT Plot
    ax3 = axes[1, 0]
    ax3.plot(levels, avg_ttfts, marker='^', linestyle='-', linewidth=2, color='orange', label='Avg TTFT')
    ax3.set_title("Concurrency vs Time To First Token")
    ax3.set_xlabel("Concurrent Requests")
    ax3.set_ylabel("Average TTFT (seconds)")
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend()
    
    # 4. Success Rate Plot
    ax4 = axes[1, 1]
    ax4.plot(levels, [sr * 100 for sr in success_rates], marker='d', linestyle='-', linewidth=2, color='purple', label='Success Rate')
    ax4.set_title("Concurrency vs Success Rate")
    ax4.set_xlabel("Concurrent Requests")
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_ylim(-5, 105)
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(args.out, dpi=300)
    print(f"Graph saved to {args.out}")

if __name__ == '__main__':
    asyncio.run(main())
