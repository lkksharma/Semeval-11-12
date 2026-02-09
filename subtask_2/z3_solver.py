#!/usr/bin/env python3
"""
Logic-LM Solver for SemEval Task 11
Translates Natural Language Syllogisms -> Z3 Python Code -> Validity
"""
import os
import sys
import re
import subprocess
import tempfile
import time
import google.generativeai as genai

class Z3SyllogismSolver:
    def __init__(self, check_install=True):
        if check_install:
             self._check_z3_installed()
             
    def _check_z3_installed(self):
        try:
            import z3
        except ImportError:
            print("WARNING: z3-solver not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "z3-solver"])

    def create_z3_prompt(self, syllogism: str) -> str:
        prompt = f"""You are a Logic-Z3 Expert.
Goal: Prove validity of this syllogism by converting it to a Z3 Python script.

SYLLOGISM:
"{syllogism}"

INSTRUCTIONS:
1. Define a Sort/Enum for the entities (Universe).
2. Define predicates for the Terms (Subject, Predicate, Middle).
3. Add constraints for the Premises.
4. To check VALIDITY, we use Proof by Contradiction:
   - Assert Premises.
   - Assert NOT(Conclusion).
   - Check Satisfiability.
   - If UNSAT => VALID (Contradiction found, so Conclusion must follow).
   - If SAT => INVALID (Counter-example found).

CODE TEMPLATE:
```python
from z3 import *

# 1. Declare Sorts and Functions
# Create a generic universe or specific object/sort
U = DeclareSort('U')
x = Const('x', U)

# Identify Terms: e.g., greek(x), mortal(x), man(x)
# Extract these from the syllogism.
# ... [Your Definitions] ...

sl = Solver()

# 2. Translate Premises
# e.g. "All Men are Mortal" -> ForAll([x], Implies(Man(x), Mortal(x)))
# e.g. "Some Men are Greek" -> Exists([x], And(Man(x), Greek(x)))
# ... [Your Translation] ...

# 3. Translate Conclusion (NEGATED)
# Conclusion: "Some Greek are Mortal"
# Negation: "No Greek are Mortal" -> ForAll([x], Implies(Greek(x), Not(Mortal(x))))
# ... [Your Negated Conclusion] ...

# 4. Check
print("CHECKING...")
if sl.check() == unsat:
    print("RESULT: VALID")
else:
    print("RESULT: INVALID")
```

IMPORTANT rules:
- Code must be complete and runnable.
- Do not use 'pass' or placeholders.
- Handle "existential import" if needed (Aristotelian view often implies existence for 'All', but standard FOL does not. Follow standard FOL unless 'All' clearly implies existence).
- OUTPUT ONLY THE CODE inside ```python ... ``` blocks.
"""
        return prompt

    def generate_code(self, model, syllogism: str) -> str:
        prompt = self.create_z3_prompt(syllogism)
        try:
            response = model.generate_content(prompt)
            text = response.text
            # Extract code block
            match = re.search(r'```python(.*?)```', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return text.strip() # Fallback
        except Exception as e:
            print(f"GenAI Error: {e}")
            return None

    def execute_code(self, code: str) -> str:
        """
        Executes the Z3 code in a separate process to avoid crashing the main thread.
        Returns: "VALID", "INVALID", or "ERROR"
        """
        if not code:
            return "ERROR"

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(code)
            
        try:
            # Run with timeout
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=5 # 5 seconds max for solver
            )
            
            output = result.stdout.strip()
            if "RESULT: VALID" in output:
                return "VALID"
            elif "RESULT: INVALID" in output:
                return "INVALID"
            else:
                # Debug info if needed
                # print(f"Z3 Output Undefined: {output}")
                return "ERROR"
                
        except subprocess.TimeoutExpired:
            return "ERROR"
        except Exception as e:
            print(f"Execution Error: {e}")
            return "ERROR"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def prove_validity(self, model, syllogism: str) -> dict:
        """
        Full pipeline: NL -> Code -> Valid/Invalid
        """
        code = self.generate_code(model, syllogism)
        if not code:
            return {"validity": False, "error": "gen_failed"}
            
        result = self.execute_code(code)
        
        is_valid = (result == "VALID")
        return {
            "validity": is_valid,
            "z3_result": result,
            "generated_code": code
        }

if __name__ == "__main__":
    # Test
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        solver = Z3SyllogismSolver()
        
        test_case = "All men are mortal. Socrates is a man. Therefore, Socrates is mortal."
        print(f"Testing: {test_case}")
        res = solver.prove_validity(model, test_case)
        print(res['z3_result'])
