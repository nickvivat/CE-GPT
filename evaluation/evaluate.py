import pandas as pd
import json
import os
from openai import OpenAI
from pathlib import Path

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Path to context-relative prompt directory
PROMPT_DIR = Path(__file__).parent.parent / "prompt" / "evaluation"

def load_prompt(filename):
    """Utility to load prompt from file system."""
    with open(PROMPT_DIR / filename, "r", encoding="utf-8") as f:
        return f.read()

def evaluate_retrieval(query, sources):
    system_prompt = load_prompt("retrieval_evaluation.md")
    user_prompt = f"[User Query]: {query}\n[Retrieved Context]: {sources}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"retrieval_score": 0, "retrieval_reason": f"Error: {str(e)}"}


def evaluate_generation(query, sources, content):
    system_prompt = load_prompt("generation_evaluation.md")
    user_prompt = f"[User Query]: {query}\n[Retrieved Context]: {sources}\n[Generated Answer]: {content}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"generation_score": 0, "generation_reason": f"Error: {str(e)}"}


print("Loading results.csv...")
df = pd.read_csv("results.csv")

total_rows = len(df)
for index, row in df.iterrows():
    print(f"Evaluating [{index+1}/{total_rows}]...")
    
    query = row['query']
    sources = row['sources']
    content = row['content']
    
    ret_eval = evaluate_retrieval(query, sources)
    df.at[index, 'retrieval_score'] = ret_eval.get('retrieval_score', 0)
    df.at[index, 'retrieval_reason'] = ret_eval.get('retrieval_reason', '')
    
    gen_eval = evaluate_generation(query, sources, content)
    df.at[index, 'generation_score'] = gen_eval.get('generation_score', 0)
    df.at[index, 'generation_reason'] = gen_eval.get('generation_reason', '')

retrieval_accuracy = (df['retrieval_score'].sum() / total_rows) * 100
generation_accuracy = (df['generation_score'].sum() / total_rows) * 100

print("\n=== Two-Step Evaluation Complete ===")
print(f"Retrieval Accuracy (Context Relevance): {retrieval_accuracy:.2f}%")
print(f"Generation Accuracy (Answer Quality & Domain Safety): {generation_accuracy:.2f}%")

df.to_csv("two_step_evaluated_results.csv", index=False, encoding='utf-8-sig')
print("Results saved to two_step_evaluated_results.csv")