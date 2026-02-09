#!/usr/bin/env python3
"""
SemEval-2026 Task 11 Subtask 2: Retrieval + Neuro-Symbolic Classification
Goal: Predict validity AND identify relevant premises (indices).
Approach: 
1. Gemini 2.0 Flash for RETRIEVAL (find the 2 most relevant premises).
2. SymbolicSyllogismEngine (Subtask 1) for VALIDITY (A/E/I/O + Figure -> check_validity).

Optimization: ThreadPoolExecutor for concurrent API calls.
"""

import json
import argparse
import os

# Prevent PyTorch/Tokenizers from spawning threads inside ThreadPool workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import time
import sys
import zipfile
import threading
from tqdm import tqdm
import google.generativeai as genai

# Import local modules (Self-contained for A100 deployment)
from symbolic_syllogism_engine import SymbolicSyllogismEngine, extract_structure_with_gemini, rule_based_extract, check_validity_symbolic, anonymize_syllogism, T5SyllogismParser
from collections import Counter

import spacy
# Reverted to simple imports (spacy removed for stability)
try:
    nlp = None # spacy.load("en_core_web_sm")
except:
    nlp = None

# Reverted: Removed robust_anonymize and create_logic_prompt to restore baseline accuracy.

class EnsembleSymbolicEngine(SymbolicSyllogismEngine):
    """
    Extends SymbolicSyllogismEngine to use Ensemble Voting (Gemini + T5 + Rule).
    """
    def __init__(self, gemini_model=None, t5_path="t5_syllogism_parser"):
        t5_parser = None
        
        # User specified path
        user_ckpt = "../../t5_syllogism_parser/checkpoint-5"
        
        # Priority order: User specified, then standard locations
        checkpoints = [user_ckpt, os.path.join(t5_path, "checkpoint-5"), os.path.join(t5_path, "checkpoint-2"), os.path.join(t5_path, "checkpoint-1")]
        
        for ckpt_path in checkpoints:
            if os.path.exists(ckpt_path) and (os.path.exists(os.path.join(ckpt_path, "config.json")) or os.path.exists(os.path.join(ckpt_path, "tokenizer.json"))):
                 print(f"Using T5 Checkpoint: {ckpt_path}")
                 t5_path = ckpt_path
                 break
        
        # Check if user provided the trained T5 model
        if os.path.exists(t5_path) and (os.path.exists(os.path.join(t5_path, "config.json")) or os.path.exists(os.path.join(t5_path, "tokenizer.json"))):
            try:
                print(f"Loading T5 Parser from {t5_path}...")
                t5_parser = T5SyllogismParser(t5_path)
            except Exception as e:
                print(f"Warning: Could not load T5 parser: {e}")
        else:
            print("Notice: T5 parser not found. Ensemble will effectively be Gemini + Rule.")
            
        super().__init__(gemini_model=gemini_model, t5_parser=t5_parser)
        
    def predict_validity_ensemble(self, syllogism: str, anonymize: bool = True) -> tuple:
        """
        Ensemble prediction: Votes on STRUCTURE (Mood/Figure).
        Reverted to majority vote on full tuple (Mood+Figure) for stability.
        """
        processed_syllogism = syllogism
        if anonymize:
            processed_syllogism = anonymize_syllogism(syllogism)
            
        votes = []
        structures = []
        
        # 1. T5 Parser
        if self.t5_parser:
            try:
                s_t5 = self.t5_parser.extract(processed_syllogism)
                if s_t5:
                    key = (s_t5['premise1_type'], s_t5['premise2_type'], s_t5['conclusion_type'], s_t5['figure'])
                    votes.append(key)
                    structures.append(('T5', s_t5))
            except Exception as e:
                print(f"T5 ensemble error: {e}")

        # 2. Gemini Parser
        if self.gemini_model:
            s_gem = extract_structure_with_gemini(processed_syllogism, self.gemini_model)
            if s_gem:
                key = (s_gem['premise1_type'], s_gem['premise2_type'], s_gem['conclusion_type'], s_gem['figure'])
                votes.append(key)
                structures.append(('Gemini', s_gem))
                
        # 3. Rule-based Parser (Tie-breaker)
        s_rule = rule_based_extract(processed_syllogism)
        if s_rule:
             key = (s_rule['premise1_type'], s_rule['premise2_type'], s_rule['conclusion_type'], s_rule['figure'])
             votes.append(key)
             structures.append(('Rule', s_rule))
             
        if not votes:
            # Fallback to invalid if no one could parse it
            return False, {'error': 'no_votes', 'anonymized': processed_syllogism}
            
        # Majority Vote
        vote_counts = Counter(votes)
        # most_common returns [(elem, count), ...]
        winner_key = vote_counts.most_common(1)[0][0] 
        
        mood = (winner_key[0], winner_key[1], winner_key[2])
        figure = winner_key[3]
        
        # Validity
        validity = check_validity_symbolic(mood, figure, self.use_existential_import)
        
        return validity, {
            'mood': mood,
            'figure': figure,
            'form': f"{mood[0]}{mood[1]}{mood[2]}-{figure}",
            'votes': str(votes),
            'winner': str(winner_key),
            'anonymized_text': processed_syllogism
        }


# ============================================================================
# CONFIGURATION
# ============================================================================
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)

# Global lock removed (not needed for sequential main thread processing)

# ============================================================================
# HELPER: SENTENCE SPLITTER
# ============================================================================
def split_sentences(text):
    """
    Split text into sentences to index them.
    Simple heuristic: Split by '.', '?', '!' followed by space.
    """
    # Regex lookbehind to split by punctuation followed by space or end of string
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]


# ============================================================================
# GEMINI RETRIEVAL PROMPT (Reasoning-Guided / Chain-of-Thought)
# ============================================================================
def create_retrieval_prompt(sentences):
    """
    Ask the model to identify the 2 premises that form a syllogism with the conclusion.
    Uses Chain-of-Thought to ensure minimal hallucinations.
    """
    conclusion = sentences[-1]
    premises = sentences[:-1]
    
    numbered_text = ""
    for i, p in enumerate(premises):
        numbered_text += f"{i+1}. {p}\n"
    
    prompt = f"""You are a logician tackling a retrieval task.
Task: Identify the TWO premises from the list that are most relevant to the Conclusion.
Usually, these two premises share a 'Middle Term' with each other and connect the Subject and Predicate of the Conclusion.

CONTEXT:
{numbered_text}
CONCLUSION: {conclusion}

INSTRUCTIONS:
1. Analyze the Conclusion to identify the Subject (S) and Predicate (P).
2. Look for a premise containing S.
3. Look for a premise containing P.
4. Verify they share a Middle Term (M).
5. Return ONLY the indices of these 2 premises in JSON.

=== YOUR TURN ===
Respond with ONLY this JSON:
{{
    "reasoning": "<short explanation>",
    "relevant_premises": [<index1>, <index2>]
}}
"""
    return prompt

# ============================================================================
# PROCESS SINGLE ITEM
# ============================================================================
def process_sample(item, retrieval_model, symbolic_engine):
    uid = item['id']
    text = item['syllogism']
    
    # 1. Split Sentences
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return {"id": uid, "validity": False, "relevant_premises": []}
        
    conclusion = sentences[-1]
        
    # 2. Retrieve Indices (Gemini with CoT)
    prompt = create_retrieval_prompt(sentences)
    
    max_retries = 8
    base_delay = 2
    
    indices = []
    validity = False
    
    # Rate Limit Handling
    for attempt in range(max_retries):
        try:
            response = retrieval_model.generate_content(prompt)
            raw = response.text
            # Clean Markdown
            raw = re.sub(r'```json\s*', '', raw)
            raw = re.sub(r'```', '', raw).strip()
            
            result = json.loads(raw)
            
            indices = result.get("relevant_premises", [])
            
            # Convert 1-based to 0-based
            indices_0b = [int(x) - 1 for x in indices if isinstance(x, (int, str)) and str(x).isdigit() and int(x) > 0]
            
            # Filter out of bounds
            final_indices = [ix for ix in indices_0b if 0 <= ix < len(sentences)-1]
            
            # Logic Valid Check: Need exactly 2 premises usually
            if len(final_indices) >= 2:
                # Take top 2
                final_indices = final_indices[:2]
                p1 = sentences[final_indices[0]]
                p2 = sentences[final_indices[1]]
                
                # Construct clean syllogism string
                syllogism_text = f"{p1} {p2} Therefore, {conclusion}"
                
                # 4. Symbolic Context Check using Z3 Engine
                # We use the updated SymbolicSyllogismEngine which now has Z3 inside predict_validity
                is_valid, metadata = symbolic_engine.predict_validity(syllogism_text, use_fallback=True, anonymize=False)
                validity = is_valid
            else:
                validity = False
            
            return {
                "id": uid,
                "validity": validity,
                "relevant_premises": final_indices
            }
            
        except Exception as e:
            if "429" in str(e) or "Resource exhausted" in str(e) or "503" in str(e):
                sleep_time = base_delay * (2 ** attempt)
                sleep_time = min(sleep_time, 60)
                time.sleep(sleep_time)
            else:
                print(f"Error on {uid}: {e}")
                break
    
    # Failure fallback
    return {"id": uid, "validity": False, "relevant_premises": []}

# ============================================================================
# ENGINE
# ============================================================================
def run_subtask2(test_file, output_file, max_workers=1):
    print(f"Loading data from {test_file}...")
    with open(test_file, 'r') as f:
        data = json.load(f)
        
    # 1. Models
    # Configure API
    # Using the standard genai configure from global scope logic
    
    retrieval_model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Initialize Symbolic Engine with Z3
    # We use the BASE engine now, assuming it has Z3 integrated
    print("Initializing Symbolic Engine with Logic-LM (Z3)...")
    symbolic_engine = SymbolicSyllogismEngine(gemini_model=retrieval_model, use_z3=True)

    # RESUME LOGIC
    submission = []
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                submission = json.load(f)
                processed_ids = {item['id'] for item in submission}
            print(f"Resuming from {len(submission)} existing samples...")
        except json.JSONDecodeError:
            print("Output file corrupted, starting fresh.")
            submission = []
            
    # Filter items
    items_to_process = [item for item in data if item['id'] not in processed_ids]
    print(f"Processing {len(items_to_process)} samples...")
    
    # Sequential Processing for maximum stability/robustness
    # (Thread pools often cause issues with Z3/complex libs)
    for i, item in enumerate(tqdm(items_to_process)):
        try:
            result = process_sample(item, retrieval_model, symbolic_engine)
            submission.append(result)
            
            # Save periodically
            if len(submission) % 10 == 0:
                with open(output_file, 'w') as f:
                    json.dump(submission, f, indent=2)
                    
        except Exception as e:
            print(f"Error processing {item['id']}: {e}")

    # Final Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)
        
    # Zip
    zip_name = output_file.replace(".json", ".zip")
    with zipfile.ZipFile(zip_name, 'w') as zf:
        zf.write(output_file, arcname="predictions.json")
    print(f"Created {zip_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output", default="predictions.json")
    # Workers argument kept for compatibility but ignored for sequential safety
    parser.add_argument("--workers", type=int, default=1, help="Ignored (Running Sequential for Safety)")
    args = parser.parse_args()
    
    run_subtask2(args.test_data, args.output, args.workers)
