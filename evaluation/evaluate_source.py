import pandas as pd
import json
import ast
import os
from openai import OpenAI
from pathlib import Path

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Path to context-relative prompt directory
PROMPT_DIR = Path(__file__).parent.parent / "prompt" / "evaluation"

def load_prompt(filename):
    """Utility to load prompt from file system."""
    with open(PROMPT_DIR / filename, "r", encoding="utf-8") as f:
        return f.read()

def evaluate_source(query, source_content):
    """
    Calls OpenAI to evaluate if a single source is a True Positive (relevant) 
    or False Positive (irrelevant) for the given query.
    """
    prompt_template = load_prompt("source_evaluation.md")
    user_prompt = prompt_template.format(query=query, source_content=source_content)
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a strict evaluator for search relevance. Output only 'TP' or 'FP'."},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during API call: {e}")
        return "ERROR"

def main():
    input_csv = "results.csv"
    output_csv = "evaluated_results.csv"
    
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    evaluated_rows = []

    for index, row in df.iterrows():
        case_id = row.get("case_id", f"row_{index}")
        query = row.get("query", "")
        sources_str = row.get("sources", "[]")
        
        if pd.isna(sources_str):
            sources_str = "[]"
        
        try:
            sources_list = json.loads(sources_str)
        except json.JSONDecodeError:
            try:
                sources_list = ast.literal_eval(sources_str)
            except (SyntaxError, ValueError):
                print(f"Could not parse sources for {case_id}. Skipping.")
                continue

        print(f"\nEvaluating Query: {case_id} ({len(sources_list)} sources)")

        for source_idx, source in enumerate(sources_list):
            source_content = source.get("content", "")
            chunk_id = source.get("chunk_id", f"chunk_{source_idx}")
            
            classification = evaluate_source(query, source_content)
            
            evaluated_rows.append({
                "case_id": case_id,
                "query": query,
                "source_index": source_idx,
                "chunk_id": chunk_id,
                "classification": classification,
                "source_content": source_content
            })
            
            print(f"  -> Source {source_idx}: {classification}")

    result_df = pd.DataFrame(evaluated_rows)
    result_df.to_csv(output_csv, index=False)
    print(f"\nEvaluation complete! Results saved to {output_csv}")

if __name__ == "__main__":
    main()