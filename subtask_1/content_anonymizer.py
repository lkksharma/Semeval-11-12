"""
Content Anonymization for SemEval-2026 Task 11

Replaces semantic content (nouns, entities) with random tokens before
sending to Gemini, forcing purely structural parsing. This eliminates
content bias (TCE → 0) by design.
"""

import re
import random
import string
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from parsers.entity_identity_parser import EntityIdentityParser, Quantifier
from engines.syllogism_validator import SyllogismValidator

# Try importing spacy for better NER, fallback to regex
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False


@dataclass
class AnonymizedSyllogism:
    """Syllogism with anonymized content"""
    original: str
    anonymized: str
    entity_map: Dict[str, str]  # token -> original entity
    reverse_map: Dict[str, str]  # original -> token


def generate_random_token(length: int = 3) -> str:
    """Generate a random alphanumeric token"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def anonymize_with_regex(syllogism: str) -> AnonymizedSyllogism:
    """
    Anonymize syllogism using regex pattern matching.
    
    Strategy:
    1. Identify quantifier patterns (All, No, Some)
    2. Extract noun phrases between quantifiers and verbs
    3. Replace with random tokens
    """
    # Split into sentences
    sentences = re.split(r'[.;]', syllogism)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Patterns for extracting entities
    # "All X are Y" / "No X is Y" / "Some X are Y" / "Some X are not Y"
    patterns = [
        r'\b(All|Every|Each)\s+(.+?)\s+(are|is)\s+(.+)',  # Universal affirmative
        r'\b(No|None of the)\s+(.+?)\s+(are|is)\s+(.+)',  # Universal negative
        r'\b(Some|A few|Certain)\s+(.+?)\s+(are|is)\s+not\s+(.+)',  # Particular negative
        r'\b(Some|A few|Certain|There exist)\s+(.+?)\s+(are|is|that are)\s+(.+)',  # Particular affirmative
    ]
    
    # Extract all unique entities
    entities = set()
    for sent in sentences:
        sent_clean = sent.replace("Therefore, ", "").replace("Hence, ", "").strip()
        for pattern in patterns:
            match = re.search(pattern, sent_clean, re.IGNORECASE)
            if match:
                groups = match.groups()
                # Subject is group 2, predicate is group 4
                if len(groups) >= 4:
                    subj = groups[1].strip().lower()
                    pred = groups[3].strip().lower()
                    # Clean up
                    subj = re.sub(r'^(a |an |the )', '', subj)
                    pred = re.sub(r'^(a |an |the )', '', pred)
                    pred = re.sub(r'[.,;]$', '', pred)
                    if subj:
                        entities.add(subj)
                    if pred:
                        entities.add(pred)
                break
    
    # Create mapping
    entity_list = sorted(entities, key=lambda x: -len(x))  # Longer first for replacement
    entity_map = {}  # token -> original
    reverse_map = {}  # original -> token
    
    for i, entity in enumerate(entity_list):
        token = f"ENTITY{i+1}"
        entity_map[token] = entity
        reverse_map[entity] = token
    
    # Replace in text
    anonymized = syllogism
    for entity, token in sorted(reverse_map.items(), key=lambda x: -len(x[0])):
        # Case-insensitive replacement
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        anonymized = pattern.sub(token, anonymized)
    
    return AnonymizedSyllogism(
        original=syllogism,
        anonymized=anonymized,
        entity_map=entity_map,
        reverse_map=reverse_map
    )


def anonymize_with_spacy(syllogism: str) -> AnonymizedSyllogism:
    """Anonymize using spaCy NER and noun chunking"""
    if not SPACY_AVAILABLE:
        return anonymize_with_regex(syllogism)
    
    doc = nlp(syllogism)
    
    # Collect all noun chunks and named entities
    entities = set()
    for chunk in doc.noun_chunks:
        # Remove determiners
        text = chunk.text.lower()
        text = re.sub(r'^(all |every |no |some |a |an |the )', '', text)
        if text and len(text) > 1:
            entities.add(text)
    
    for ent in doc.ents:
        entities.add(ent.text.lower())
    
    # Create mapping
    entity_list = sorted(entities, key=lambda x: -len(x))
    entity_map = {}
    reverse_map = {}
    
    for i, entity in enumerate(entity_list):
        token = f"ENTITY{i+1}"
        entity_map[token] = entity
        reverse_map[entity] = token
    
    # Replace
    anonymized = syllogism
    for entity, token in sorted(reverse_map.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        anonymized = pattern.sub(token, anonymized)
    
    return AnonymizedSyllogism(
        original=syllogism,
        anonymized=anonymized,
        entity_map=entity_map,
        reverse_map=reverse_map
    )


def anonymize_simple(syllogism: str) -> AnonymizedSyllogism:
    """
    Simple but robust anonymization:
    Replace ALL nouns (detected by common patterns) with tokens.
    """
    # Split by statements
    parts = re.split(r'(?<=[.;])\s*', syllogism)
    parts = [p.strip() for p in parts if p.strip()]
    
    # Common quantifier patterns
    quantifier_patterns = [
        r'^(Therefore,?\s*)?(All|Every|Each|Any)\s+',
        r'^(Therefore,?\s*)?(No|None of the)\s+',
        r'^(Therefore,?\s*)?(Some|A few|Certain|Many)\s+',
    ]
    
    entities = []
    
    for part in parts:
        # Remove "Therefore" prefix
        clean = re.sub(r'^Therefore,?\s*', '', part, flags=re.IGNORECASE)
        
        # Pattern: QUANTIFIER + SUBJECT + VERB + PREDICATE
        # Try to extract subject and predicate
        
        # Universal: "All X are Y" / "No X is Y"
        match = re.match(r'(All|Every|Each|No|None of the|Some|A few|Certain|Many)\s+(.+?)\s+(are|is)\s+(not\s+)?(.+)', clean, re.IGNORECASE)
        if match:
            subj = match.group(2).strip()
            pred = match.group(5).strip().rstrip('.,;')
            
            # Clean articles
            subj = re.sub(r'^(a |an |the )', '', subj, flags=re.IGNORECASE)
            pred = re.sub(r'^(a |an |the )', '', pred, flags=re.IGNORECASE)
            
            if subj and subj.lower() not in [e.lower() for e in entities]:
                entities.append(subj)
            if pred and pred.lower() not in [e.lower() for e in entities]:
                entities.append(pred)
    
    # Sort by length (longer first) for replacement
    entities = sorted(set(entities), key=lambda x: -len(x))
    
    # Create mapping with simple tokens
    entity_map = {}
    reverse_map = {}
    tokens = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]
    
    for i, entity in enumerate(entities[:5]):  # Max 5 entities (typically 3)
        token = tokens[i] if i < len(tokens) else f"ENT{i}"
        entity_map[token] = entity
        reverse_map[entity.lower()] = token
    
    # Replace in text (case-insensitive)
    anonymized = syllogism
    for entity, token in sorted(reverse_map.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
        anonymized = pattern.sub(token, anonymized)
    
    return AnonymizedSyllogism(
        original=syllogism,
        anonymized=anonymized,
        entity_map=entity_map,
        reverse_map=reverse_map
    )


class AnonymizedParser:
    """Parser that anonymizes content before sending to Gemini"""
    
    EXTRACTION_PROMPT = """You are a precise logical parser. Given a syllogism with anonymized entities (ALPHA, BETA, GAMMA, etc.), extract the logical structure.

For each of the 3 statements, identify:
1. Quantifier type:
   - A = "All X are Y" (Universal Affirmative)
   - E = "No X is Y" (Universal Negative)
   - I = "Some X are Y" (Particular Affirmative)
   - O = "Some X are not Y" (Particular Negative)
2. Subject entity (ALPHA, BETA, or GAMMA)
3. Predicate entity (ALPHA, BETA, or GAMMA)

IMPORTANT: Entities are already anonymized. Focus ONLY on logical structure.

Respond with ONLY valid JSON:
{
  "entities": ["ALPHA", "BETA", "GAMMA"],
  "premise1": {"quantifier": "A/E/I/O", "subject": "ALPHA/BETA/GAMMA", "predicate": "ALPHA/BETA/GAMMA"},
  "premise2": {"quantifier": "A/E/I/O", "subject": "ALPHA/BETA/GAMMA", "predicate": "ALPHA/BETA/GAMMA"},
  "conclusion": {"quantifier": "A/E/I/O", "subject": "ALPHA/BETA/GAMMA", "predicate": "ALPHA/BETA/GAMMA"}
}

Anonymized syllogism:
"""

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = None
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(model_name)
            self.genai = genai
        except Exception as e:
            print(f"Warning: Could not initialize Gemini: {e}")
    
    def parse(self, syllogism: str) -> Tuple[dict, AnonymizedSyllogism]:
        """Parse syllogism with anonymization"""
        # Anonymize
        anon = anonymize_simple(syllogism)
        
        # Send to Gemini
        prompt = self.EXTRACTION_PROMPT + anon.anonymized
        
        response = self.model.generate_content(
            prompt,
            generation_config=self.genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=500,
            )
        )
        
        # Parse JSON response
        text = response.text
        # Find JSON
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            parsed = json.loads(text[start:end+1])
        else:
            raise ValueError(f"Could not parse JSON: {text[:200]}")
        
        return parsed, anon


def test_anonymization():
    """Test the anonymization on sample syllogisms"""
    samples = [
        "All cats are mammals. All mammals are animals. Therefore, all cats are animals.",
        "No fish is a mammal. All whales are mammals. Therefore, no whale is a fish.",
        "Some birds are not fliers. All penguins are birds. Therefore, some penguins are not fliers.",
    ]
    
    print("Testing Content Anonymization")
    print("=" * 60)
    
    for i, syl in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Original:   {syl}")
        anon = anonymize_simple(syl)
        print(f"  Anonymized: {anon.anonymized}")
        print(f"  Entities:   {anon.entity_map}")


if __name__ == "__main__":
    test_anonymization()
