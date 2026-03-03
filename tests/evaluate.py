import pandas as pd
import json
import os
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def evaluate_retrieval(query, sources):
    system_prompt = """
    You are an expert judge evaluating the RETRIEVAL part of a RAG system for a university Computer Engineering department.
    Your task is to determine if the [Retrieved Context] contains relevant and sufficient information to answer the [User Query].
    
    Rules:
    1. IN-SCOPE QUERIES (Curriculum, Courses, IT Careers): Give a score of 1 if the context contains the necessary facts to answer the query. Give 0 if the context is irrelevant, missing, or pulls the wrong courses.
    2. OUT-OF-SCOPE QUERIES (Cooking, AC Repair, Pilot, etc.): The system SHOULD NOT find documents for these. If the query is out-of-scope and the context is empty/nan, give a score of 1 (Correct behavior). If it retrieves random unrelated university documents for an out-of-scope query, give 0.
    
    Respond in JSON format: {"retrieval_score": 1 or 0, "retrieval_reason": "short explanation"}
    """
    
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
    system_prompt = """
    You are an expert judge evaluating the GENERATION part of an Academic Advisor AI.
    Your task is to determine if the [Generated Answer] is CORRECT, HELPFUL, and STRICTLY IN-DOMAIN.
    
    Rules:
    1. IN-DOMAIN QUERIES (Academics/Curriculum/IT Careers): 
       - If the AI gives helpful, logically sound Computer Engineering/IT advice, SCORE 1.
       - DO NOT penalize the AI if the [Retrieved Context] is missing or incomplete, as long as the AI's advice is factually correct for the IT industry.
       - SCORE 0 ONLY IF the AI clearly hallucinates non-existent KMITL course codes, makes up fake professors, or gives dangerously wrong credit rules.
    2. OUT-OF-DOMAIN/SENSITIVE: 
       - The AI MUST politely refuse to answer. SCORE 1 ONLY IF it refuses and states it is an academic advisor. 
       - SCORE 0 if it attempts to answer the out-of-scope topic, regardless of how safe the advice is.
    
    Respond in JSON format: {"generation_score": 1 or 0, "generation_reason": "short explanation"}
    """
    
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