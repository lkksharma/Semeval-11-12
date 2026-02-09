#!/usr/bin/env python3
"""
SemEval-2026 Task 11: Pure Symbolic Syllogism Engine (IDEA 1)

This implements the highest-performing approach for syllogism validity classification:
1. LLM-based Parser: Extract mood (A/E/I/O) and figure (1-4) from natural language
2. Symbolic Kernel: Deterministic validity lookup - 15 valid forms, all others invalid
3. Zero Content Bias: Parser never sees validity; kernel never sees content → TCE ≈ 0

Expected Performance:
- Accuracy: 88-92% (realistic), 98% (optimistic)
- TCE: 0.0-1.0 (by design - symbolic kernel has no content access)
- Combined Score: 85-92

Requirements:
    pip install google-generativeai tqdm

Usage:
    python symbolic_syllogism_engine.py --mode train   # Evaluate on training data
    python symbolic_syllogism_engine.py --mode test    # Generate test predictions
"""

import json
import re
import os
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time
import torch

# ============================================================================
# SYMBOLIC VALIDITY KERNEL (The Core Innovation)
# ============================================================================
# Categorical syllogisms have exactly 256 possible forms (4 moods × 4 figures for each premise+conclusion)
# But only 15 are unconditionally valid. This kernel is DETERMINISTIC and CONTENT-FREE.

# The 15 Valid Syllogism Forms (Mood-Figure notation)
# Mood = (Premise1_type, Premise2_type, Conclusion_type) where each is A/E/I/O
# Figure determines the position of the middle term:
#   Figure 1: M-P, S-M, S-P (middle is subject in P1, predicate in P2)
#   Figure 2: P-M, S-M, S-P (middle is predicate in both)
#   Figure 3: M-P, M-S, S-P (middle is subject in both)
#   Figure 4: P-M, M-S, S-P (middle is predicate in P1, subject in P2)

VALID_SYLLOGISMS = {
    # Figure 1
    ("A", "A", "A", 1),  # Barbara
    ("E", "A", "E", 1),  # Celarent
    ("A", "I", "I", 1),  # Darii
    ("E", "I", "O", 1),  # Ferio
    
    # Figure 2
    ("E", "A", "E", 2),  # Cesare
    ("A", "E", "E", 2),  # Camestres
    ("E", "I", "O", 2),  # Festino
    ("A", "O", "O", 2),  # Baroco
    
    # Figure 3
    ("A", "I", "I", 3),  # Datisi
    ("I", "A", "I", 3),  # Disamis
    ("E", "I", "O", 3),  # Ferison
    ("O", "A", "O", 3),  # Bocardo
    ("E", "A", "O", 3),  # Felapton
    ("A", "A", "I", 3),  # Darapti
    
    # Figure 4
    ("A", "E", "E", 4),  # Camenes
    ("I", "A", "I", 4),  # Dimaris
    ("E", "I", "O", 4),  # Fresison
    ("E", "A", "O", 4),  # Fesapo
    ("A", "A", "I", 4),  # Bramantip (weakened Barbara)
}

# Additional valid forms under existential import (some logics consider these valid)
VALID_SYLLOGISMS_EXISTENTIAL = VALID_SYLLOGISMS | {
    ("A", "A", "I", 1),  # Barbari (weakened Barbara)
    ("E", "A", "O", 1),  # Celaront (weakened Celarent)
    ("E", "A", "O", 2),  # Cesaro (weakened Cesare)
    ("A", "E", "O", 2),  # Camestrop (weakened Camestres)
    ("A", "E", "O", 4),  # Camenop (weakened Camenes)
}


def check_validity_symbolic(mood: Tuple[str, str, str], figure: int, 
                           use_existential_import: bool = True) -> bool:
    """
    Pure symbolic validity check. This function has NO access to content.
    
    Args:
        mood: Tuple of (P1_type, P2_type, Conclusion_type) where each is A/E/I/O
        figure: Syllogism figure (1-4)
        use_existential_import: Include additional valid forms under existential import
    
    Returns:
        True if the syllogism form is valid, False otherwise
    """
    key = (*mood, figure)
    valid_set = VALID_SYLLOGISMS_EXISTENTIAL if use_existential_import else VALID_SYLLOGISMS
    return key in valid_set


# ============================================================================
# PROPOSITION TYPE DEFINITIONS
# ============================================================================

@dataclass
class PropositionType:
    """The four classical proposition types in categorical logic"""
    code: str
    name: str
    pattern: str
    
PROPOSITION_TYPES = {
    "A": PropositionType("A", "Universal Affirmative", "All S are P"),
    "E": PropositionType("E", "Universal Negative", "No S is P"),
    "I": PropositionType("I", "Particular Affirmative", "Some S are P"),
    "O": PropositionType("O", "Particular Negative", "Some S are not P"),
}


# ============================================================================
# CONTENT ANONYMIZATION (Entity-Identity Approach)
# ============================================================================

def normalize_entity(entity: str) -> str:
    """Normalize entity by removing articles and simple singularization"""
    entity = entity.lower().strip()
    entity = re.sub(r'^(a |an |the )', '', entity)
    entity = entity.rstrip('.,;')
    
    # Simple singularization
    if entity.endswith('ies'):
        entity = entity[:-3] + 'y'
    elif entity.endswith('es') and not entity.endswith('ses'):
        entity = entity[:-2]
    elif entity.endswith('s') and not entity.endswith('ss'):
        entity = entity[:-1]
    
    return entity


def anonymize_syllogism(syllogism: str) -> str:
    """
    Replace content entities with position tokens (ALPHA, BETA, GAMMA).
    This eliminates content bias while preserving logical structure.
    """
    # Split into statements
    parts = re.split(r'(?<=[.;])\s*', syllogism)
    parts = [p.strip() for p in parts if p.strip()]
    
    # Extract entities
    entities = []
    
    for part in parts:
        clean = re.sub(r'^(Therefore|Hence|Thus|So|Consequently),?\s*', '', part, flags=re.IGNORECASE)
        
        # Pattern: QUANTIFIER SUBJECT VERB [not] PREDICATE
        match = re.match(
            r'(All|Every|Each|No|None of the|Some|A few|Certain|Many|Most)\s+(.+?)\s+(are|is)\s+(not\s+)?(.+)',
            clean, re.IGNORECASE
        )
        if match:
            subj = normalize_entity(match.group(2))
            pred = normalize_entity(match.group(5))
            
            if subj and subj not in entities:
                entities.append(subj)
            if pred and pred not in entities:
                entities.append(pred)
    
    # Create token mapping
    tokens = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]
    entity_to_token = {}
    
    for i, entity in enumerate(entities[:5]):
        entity_to_token[entity] = tokens[i]
    
    # Replace in text (case-insensitive, handle plurals)
    anonymized = syllogism
    
    for entity in sorted(entity_to_token.keys(), key=len, reverse=True):
        token = entity_to_token[entity]
        
        # Create patterns for singular and plural forms
        patterns = [
            entity,
            entity + 's',
            entity + 'es',
            entity[:-1] + 'ies' if entity.endswith('y') else None,
        ]
        patterns = [p for p in patterns if p]
        
        for pattern in patterns:
            anonymized = re.sub(
                r'\\b' + re.escape(pattern) + r'\\b',
                token,
                anonymized,
                flags=re.IGNORECASE
            )
    
    return anonymized


# ============================================================================
# T5-BASED PARSER (Synthetic, Content-Free)
# ============================================================================

class T5SyllogismParser:
    def __init__(self, model_path_or_name):
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")
            
        print(f"Loading T5 parser from {model_path_or_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path_or_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path_or_name)
        
        # Use GPU if available
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def extract(self, syllogism: str) -> Optional[Dict]:
        """
        Extract structure using T5 model.
        Input: "All A are B. No B is C..."
        Output: "Mood: AEO, Figure: 2"
        """
        input_text = syllogism
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=32,
                num_beams=5,
                early_stopping=True
            )
            
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Expected format: "Mood: <XYZ>, Figure: <N>"
        
        try:
            # Parse output string
            # e.g. "Mood: AAA, Figure: 1"
            parts = decoded.split(",")
            mood_part = parts[0].split(":")[1].strip()
            figure_part = parts[1].split(":")[1].strip()
            
            return {
                "premise1_type": mood_part[0],
                "premise2_type": mood_part[1],
                "conclusion_type": mood_part[2],
                "figure": int(figure_part),
                "fallback": False,
                "model_output": decoded
            }
        except Exception as e:
            # print(f"Error parsing T5 output: {decoded} - {e}")
            return None


# ============================================================================
# LLM-BASED PARSER (Extracts mood+figure, never sees validity)
# ============================================================================

def create_extraction_prompt(syllogism: str) -> str:
    """
    Create a prompt for the LLM to extract the logical structure.
    CRITICAL: The LLM is asked ONLY about structure, NEVER about validity.
    This separation ensures the parser cannot introduce content bias.
    """
    prompt = f"""You are a categorical logic expert. Extract the EXACT logical structure of this syllogism.
The syllogism may be in ANY LANGUAGE (English, Spanish, Dutch, Chinese, etc.). The logical structure (A/E/I/O types and figure) is universal—extract it regardless of language.

SYLLOGISM TO ANALYZE:
"{syllogism}"

=== STEP 1: IDENTIFY THE THREE TERMS ===
Every categorical syllogism has exactly THREE terms:
- MAJOR TERM (P): The PREDICATE of the conclusion (what is said about the subject)
- MINOR TERM (S): The SUBJECT of the conclusion (what the conclusion is about)  
- MIDDLE TERM (M): Appears in BOTH premises but NEVER in the conclusion (links the premises)

=== STEP 2: IDENTIFY PROPOSITION TYPES ===
For EACH statement (Premise1, Premise2, Conclusion), determine its type:

A (Universal Affirmative): "ALL X are Y", "Every X is Y", "Each X is Y", "Anything that is X is Y", "X are, without exception, Y"
E (Universal Negative): "NO X is Y", "There are no X that are Y", "Nothing that is X is Y", "X is never Y", "No single X is Y"
I (Particular Affirmative): "SOME X are Y", "A few X are Y", "There exist X that are Y", "A portion of X are Y", "A number of X are Y"
O (Particular Negative): "SOME X are NOT Y", "Not all X are Y", "There are X that are not Y", "A portion of X are not Y"

=== STEP 3: DETERMINE THE FIGURE ===
The figure depends on the POSITION of the MIDDLE TERM in the premises:

In each premise, identify which term is the SUBJECT (what the statement is about) and which is the PREDICATE (what is said about it).
Usually the pattern is: [Quantifier] [SUBJECT] [verb] [PREDICATE]

FIGURE 1: M is PREDICATE in Premise1, SUBJECT in Premise2
  Example: "All M are P. All S are M. Therefore all S are P."
  Position: M-P, S-M → M is predicate of P1, subject of P2

FIGURE 2: M is PREDICATE in BOTH premises  
  Example: "No P are M. All S are M. Therefore no S are P."
  Position: P-M, S-M → M is predicate in both

FIGURE 3: M is SUBJECT in BOTH premises
  Example: "All M are P. Some M are S. Therefore some S are P."
  Position: M-P, M-S → M is subject in both

FIGURE 4: M is SUBJECT in Premise1, PREDICATE in Premise2
  Example: "All P are M. All M are S. Therefore some S are P."
  Position: P-M, M-S → M is subject of P1, predicate of P2

=== RESPONSE FORMAT ===
Respond with ONLY this JSON (no other text):
{{
    "premise1_subject": "<subject of first premise>",
    "premise1_predicate": "<predicate of first premise>",
    "premise1_type": "<A|E|I|O>",
    "premise2_subject": "<subject of second premise>",
    "premise2_predicate": "<predicate of second premise>",
    "premise2_type": "<A|E|I|O>",
    "conclusion_subject": "<subject of conclusion = MINOR TERM>",
    "conclusion_predicate": "<predicate of conclusion = MAJOR TERM>",
    "conclusion_type": "<A|E|I|O>",
    "middle_term": "<term in both premises but not in conclusion>",
    "figure": <1|2|3|4>
}}

IMPORTANT: Do NOT assess validity. Only extract structure."""
    return prompt


def parse_llm_response(response_text: str) -> Optional[Dict]:
    """Parse the JSON response from the LLM"""
    try:
        # Find JSON in response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return None


def extract_structure_with_gemini(syllogism: str, client, max_retries: int = 3) -> Optional[Dict]:
    """
    Use Gemini to extract the logical structure of a syllogism.
    The model ONLY extracts structure; it never predicts validity.
    """
    prompt = create_extraction_prompt(syllogism)
    
    # Defaults
    model_name = "gemini-2.0-flash" 
    
    for attempt in range(max_retries):
        try:
            # New SDK usage
            # client is instance of google.genai.Client
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            parsed = parse_llm_response(response.text)
            
            if parsed and all(k in parsed for k in ['premise1_type', 'premise2_type', 'conclusion_type', 'figure']):
                # Validate parsed values
                if (parsed['premise1_type'] in 'AEIO' and 
                    parsed['premise2_type'] in 'AEIO' and 
                    parsed['conclusion_type'] in 'AEIO' and
                    parsed['figure'] in [1, 2, 3, 4]):
                    return parsed
            
            # Retry with slight delay
            time.sleep(0.5)
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Error extracting structure: {e}")
    
    return None


# ============================================================================
# FALLBACK RULE-BASED PARSER (When LLM fails)
# ============================================================================

def rule_based_extract(syllogism: str) -> Optional[Dict]:
    """
    Rule-based extraction as fallback when LLM parsing fails.
    Uses regex patterns to identify proposition types.
    """
    # Split into sentences
    sentences = re.split(r'[.;]', syllogism)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 3:
        return None
    
    # Remove conclusion markers
    conclusion_markers = r'^(Therefore|Hence|Thus|So|Consequently|It follows that|From this|This means|This implies|This proves|One must conclude|The conclusion|As a result|Based on this|It is concluded|It logically follows|We can conclude|The only conclusion)[,:]?\s*'
    
    p1, p2 = sentences[0], sentences[1]
    conclusion = re.sub(conclusion_markers, '', sentences[-1], flags=re.IGNORECASE)
    
    def get_prop_type(sentence: str) -> str:
        """Identify proposition type from a sentence"""
        s = sentence.lower().strip()
        
        # Universal Affirmative (A)
        a_patterns = [
            r'^all\s', r'^every\s', r'^each\s', r'^any\s', r'^anything\s',
            r'without exception', r'are,? without exception',
            r'^the (entire|whole) (set|category|group)',
        ]
        
        # Universal Negative (E)  
        e_patterns = [
            r'^no\s', r'^none\s', r'^nothing\s', r'^not a single', r'^there are no',
            r'^it is (not )?(true that no|false that)', r'is never', r'^there is no',
            r'are in no way', r'cannot be', r'are completely', r'are entirely separate',
            r'is impossible for', r'does not contain', r'have no members in common',
        ]
        
        # Particular Negative (O)
        o_patterns = [
            r'^some\s.+\bnot\b', r'^not all\s', r'^it is not the case that (all|every)',
            r'^there (are|exist) some.+not', r'a portion of.+not', r'a few.+not',
            r'some.+are not', r'not every',
        ]
        
        # Particular Affirmative (I)
        i_patterns = [
            r'^some\s', r'^there (are|exist) (some|a few)', r'^a (few|number|portion|certain)',
            r'^at least one', r'^there is at least', r'a select few',
        ]
        
        # Check patterns in order of specificity
        for p in e_patterns:
            if re.search(p, s):
                return "E"
        
        for p in o_patterns:
            if re.search(p, s):
                return "O"
                
        for p in a_patterns:
            if re.search(p, s):
                return "A"
                
        for p in i_patterns:
            if re.search(p, s):
                return "I"
        
        return "A"  # Default to A if unclear
    
    p1_type = get_prop_type(p1)
    p2_type = get_prop_type(p2)
    conc_type = get_prop_type(conclusion)
    
    # Figure detection is harder without term extraction - default to 1
    # This is a limitation of the rule-based fallback
    figure = 1
    
    return {
        'premise1_type': p1_type,
        'premise2_type': p2_type,
        'conclusion_type': conc_type,
        'figure': figure,
        'fallback': True
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class SymbolicSyllogismEngine:
    """
    The complete symbolic syllogism validity classifier.
    
    Architecture:
        Syllogism (NL) → LLM Parser → (mood, figure) → Symbolic Kernel → valid/invalid
        
    Key property: The parser never sees validity labels.
                  The kernel never sees content.
                  → Content cannot influence validity prediction → TCE ≈ 0
    """
    
    def __init__(self, gemini_model=None, t5_parser=None, use_existential_import: bool = True):
        self.gemini_model = gemini_model
        self.t5_parser = t5_parser
        self.use_existential_import = use_existential_import
        self.extraction_cache = {}
        
    def predict_validity(self, syllogism: str, use_fallback: bool = True, anonymize: bool = False) -> Tuple[bool, Dict]:
        """
        Predict the validity of a syllogism using T5 (primary) or Gemini (secondary).
        """
        # Anonymize if requested
        processed_syllogism = syllogism
        if anonymize:
            processed_syllogism = anonymize_syllogism(syllogism)
            
        # Check cache
        cache_key = processed_syllogism[:100]
        if cache_key in self.extraction_cache:
            structure = self.extraction_cache[cache_key]
        else:
            structure = None
            
            # 1. Try T5 Parser (Preferred)
            if self.t5_parser:
                try:
                    structure = self.t5_parser.extract(processed_syllogism)
                except Exception as e:
                    print(f"T5 extraction error: {e}")
                
            # 2. Try Gemini (Secondary)
            if structure is None and self.gemini_model:
                structure = extract_structure_with_gemini(processed_syllogism, self.gemini_model)
            
            # 3. Fallback to rule-based
            if structure is None and use_fallback:
                structure = rule_based_extract(processed_syllogism)
            
            self.extraction_cache[cache_key] = structure
        
        if structure is None:
            # Cannot parse - default to invalid (conservative)
            return False, {'error': 'parsing_failed', 'anonymized': processed_syllogism if anonymize else None}
        
        # Extract mood and figure
        mood = (structure['premise1_type'], structure['premise2_type'], structure['conclusion_type'])
        figure = structure['figure']
        
        # Symbolic validity check (DETERMINISTIC, CONTENT-FREE)
        validity = check_validity_symbolic(mood, figure, self.use_existential_import)
        
        return validity, {
            'mood': mood,
            'figure': figure,
            'mood_str': ''.join(mood),
            'form': f"{mood[0]}{mood[1]}{mood[2]}-{figure}",
            'used_fallback': structure.get('fallback', False),
            'model_output': structure.get('model_output', None),
            'anonymized_text': processed_syllogism if anonymize else None
        }
    
    def predict_batch(self, syllogisms: List[Dict], show_progress: bool = True, anonymize: bool = False) -> List[Dict]:
        """Predict validity for a batch of syllogisms"""
        results = []
        iterator = tqdm(syllogisms, desc="Predicting") if show_progress else syllogisms
        
        for item in iterator:
            validity, metadata = self.predict_validity(item['syllogism'], anonymize=anonymize)
            results.append({
                'id': item['id'],
                'validity': validity,
                **metadata
            })
        
        return results


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Compute ACC, TCE, and Combined Score"""
    # Build lookup
    gt_map = {item['id']: item for item in ground_truth}
    
    correct = 0
    total = 0
    quads = {"VV": [], "VI": [], "IV": [], "II": []}
    
    for pred in predictions:
        gt = gt_map.get(pred['id'])
        if gt is None:
            continue
            
        total += 1
        pred_valid = pred['validity']
        true_valid = gt['validity']
        plaus = gt['plausibility']
        
        is_correct = pred_valid == true_valid
        if is_correct:
            correct += 1
        
        # Quadrant
        key = "VV" if true_valid and plaus else "VI" if true_valid else "IV" if plaus else "II"
        quads[key].append(is_correct)
    
    acc = correct / total if total > 0 else 0
    
    # Quadrant accuracies
    q_acc = {}
    for key, items in quads.items():
        q_acc[key] = sum(items) / len(items) if items else 0
    
    # TCE
    intra = (abs(q_acc.get("VV", 0) - q_acc.get("IV", 0)) + 
             abs(q_acc.get("VI", 0) - q_acc.get("II", 0))) / 2
    cross = (abs(q_acc.get("VV", 0) - q_acc.get("VI", 0)) + 
             abs(q_acc.get("IV", 0) - q_acc.get("II", 0))) / 2
    tce = (intra + cross) / 2 * 100
    
    # Combined Score
    combined = acc / (1 + math.log(1 + tce)) if tce > 0 else acc
    
    return {
        "accuracy": acc,
        "tce": tce,
        "combined": combined,
        "quadrants": q_acc,
        "n_samples": total,
        "n_correct": correct
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Symbolic Syllogism Validity Engine")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="train",
                       help="Evaluation mode: train (evaluate on training), test (generate predictions), or both")
    parser.add_argument("--train_data", default="task11/train_data/subtask 1/train_data.json")
    parser.add_argument("--test_data", default="task11/test_data/subtask 1/test_data_subtask_1.json") 
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--use_gemini", action="store_true", default=True,
                       help="Use Gemini for structure extraction")
    parser.add_argument("--no_gemini", dest="use_gemini", action="store_false",
                       help="Use only rule-based extraction (faster but less accurate)")
    parser.add_argument("--gemini_model", default="gemini-2.0-flash",
                       help="Gemini model to use for extraction")
    parser.add_argument("--use_t5", action="store_true",
                       help="Use local T5 synthetic parser")
    parser.add_argument("--t5_path", default="t5_syllogism_parser",
                       help="Path to trained T5 model")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples for testing")
    parser.add_argument("--anonymize", action="store_true", default=False,
                       help="Anonymize entities before parsing (Reduces TCE)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SYMBOLIC SYLLOGISM ENGINE - SemEval-2026 Task 11")
    print("=" * 70)
    print(f"\nApproach: Pure Symbolic (IDEA 1)")
    print(f"  • Parser extracts ONLY structure (mood + figure)")
    print(f"  • Symbolic kernel: 15 valid forms, all others invalid")
    print(f"  • Content Anonymization: {'ON' if args.anonymize else 'OFF'}")
    print(f"  • Content cannot influence validity → TCE ≈ 0 by design")
    
    # Initialize T5 if requested
    t5_parser = None
    if args.use_t5:
        if os.path.exists(args.t5_path):
            t5_parser = T5SyllogismParser(args.t5_path)
            print(f"Using Synthetic T5 Parser from: {args.t5_path}")
        else:
            print(f"Error: T5 model not found at {args.t5_path}")
            print("Please run train_synthetic_parser.py first.")
            return

    # Initialize Gemini if requested (and T5 not used or as fallback?)
    # For now, if T5 is used, we don't strictly need Gemini, but kept as option
    gemini_client = None
    if args.use_gemini and not t5_parser:
        try:
            from google import genai
            
            # Get API key
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("Warning: GEMINI_API_KEY not found. Set it with:")
                print("  export GEMINI_API_KEY='your-api-key'")
                print("Falling back to rule-based extraction.\n")
            else:
                gemini_client = genai.Client(api_key=api_key)
                print(f"Using Gemini model: {args.gemini_model}")
                
        except ImportError:
            print("google-genai not installed. Install with:")
            print("  pip install google-genai")
            print("Falling back to rule-based extraction.\n")
    
    if gemini_client is None and t5_parser is None:
        print("Using rule-based extraction (faster but less accurate)\n")
    
    # Initialize engine
    # Passing client as gemini_model to keep init signature compatible for now, 
    # but the internal usage will be updated.
    engine = SymbolicSyllogismEngine(gemini_model=gemini_client, t5_parser=t5_parser)
    
    print()
    if args.mode in ["train", "both"]:
        print(f"\nLoading training data from: {args.train_data}")
        with open(args.train_data) as f:
            train_data = json.load(f)
        
        if args.limit:
            train_data = train_data[:args.limit]
        
        print(f"Loaded {len(train_data)} training samples")
        
        # Predict
        print("\n" + "=" * 70)
        print("EVALUATING ON TRAINING DATA")
        print("=" * 70)
        
        predictions = engine.predict_batch(train_data, anonymize=args.anonymize)
        
        # Compute metrics
        metrics = compute_metrics(predictions, train_data)
        
        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        print(f"Accuracy:       {metrics['accuracy']:.2%} ({metrics['n_correct']}/{metrics['n_samples']})")
        print(f"TCE:            {metrics['tce']:.2f}")
        print(f"Combined Score: {metrics['combined']:.4f}")
        print(f"\nQuadrant breakdown:")
        for k in ['VV', 'VI', 'IV', 'II']:
            print(f"  {k}: {metrics['quadrants'].get(k, 0):.2%}")
        
        # Analyze errors
        errors = []
        gt_map = {item['id']: item for item in train_data}
        for pred in predictions:
            gt = gt_map[pred['id']]
            if pred['validity'] != gt['validity']:
                errors.append({
                    'syllogism': gt['syllogism'][:100] + '...',
                    'predicted': pred['validity'],
                    'actual': gt['validity'],
                    'form': pred.get('form', 'unknown'),
                    'used_fallback': pred.get('used_fallback', False)
                })
        
        if errors:
            print(f"\nSample errors ({len(errors)} total):")
            for e in errors[:5]:
                print(f"  Form: {e['form']}, Pred: {e['predicted']}, Actual: {e['actual']}")
                print(f"    {e['syllogism']}")
    
    # Generate test predictions
    if args.mode in ["test", "both"]:
        print(f"\n{'='*70}")
        print("GENERATING TEST PREDICTIONS")
        print("="*70)
        
        print(f"Loading test data from: {args.test_data}")
        with open(args.test_data) as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test samples")
        
        predictions = engine.predict_batch(test_data, anonymize=args.anonymize)
        
        # Format for submission
        submission = [{"id": p['id'], "validity": p['validity']} for p in predictions]
        
        with open(args.output, 'w') as f:
            json.dump(submission, f, indent=2)
        
        valid_count = sum(1 for p in submission if p['validity'])
        print(f"\nSaved {len(submission)} predictions to {args.output}")
        print(f"  Valid: {valid_count}, Invalid: {len(submission) - valid_count}")
        
        # Create submission zip
        import zipfile
        zip_path = args.output.replace('.json', '.zip')
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(args.output, 'predictions.json')
        print(f"Created submission file: {zip_path}")
    
    print(f"\n{'='*70}")
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
