<div align="center">

# 🧠 Neuro-Symbolic Decomposition to Mitigate Content Bias in Syllogistic Reasoning

### SemEval-2026 Task 11 — Subtasks 1 & 2

**Team lakksh**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![T5](https://img.shields.io/badge/T5--Small-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/google-t5/t5-small)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

*A neuro-symbolic system that eliminates content bias in syllogistic reasoning through architectural separation — not prompt engineering.*

</div>

---

## 📌 Overview

Large language models conflate **semantic plausibility** with **logical validity**, producing systematic content bias in deductive reasoning. This system addresses the problem through **architectural constraints**:

1. **Symbolic Bottleneck** — A neural extractor maps syllogisms to abstract `(Mood, Figure)` tuples; a deterministic kernel checks validity against 15 valid forms. The kernel never sees natural language.

2. **Amnesiac Training** — A T5-Small model is trained exclusively on synthetic nonsense syllogisms (ALPHA, BETA, GAMMA), preventing it from learning entity–validity correlations.

3. **OR-Ensemble Fusion** — Gemini (high semantic recall) and T5 (high syntactic precision) predictions are fused via logical OR, recovering false negatives without amplifying false positives.

```
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM ARCHITECTURE                         │
│                                                                 │
│  Syllogism ──► [ Gemini Parser ] ──► (Mood, Figure) ──┐        │
│             ──► [ T5 Amnesiac  ] ──► (Mood, Figure) ──┼──► OR  │
│             ──► [ Rule-Based   ] ──► (Mood, Figure) ──┘   │    │
│                                                           ▼    │
│                                          ┌──────────────────┐  │
│                                          │ Symbolic Kernel   │  │
│                                          │ (15 valid forms)  │  │
│                                          │ Zero parameters   │  │
│                                          └────────┬─────────┘  │
│                                                   ▼            │
│                                            VALID / INVALID     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Results

| Subtask | Accuracy | TCE ↓ | Combined |
|---------|----------|-------|----------|
| **Subtask 1** (Binary Validity) | **97.38%** | **3.10** | **31.43** |
| **Subtask 2** (Retrieval + Validity) | 85.94 | 13.18 | 24.46 |

> **Key insight:** Content robustness is achieved via design constraints, not prompting. The NL→Z3 theorem proving approach was rejected because semantic information leaks during translation — *where* symbolic reasoning is introduced matters as much as *whether* it is used.

---

## 📁 Repository Structure

```
Neuro-Symbolic Decomposition/
├── README.md
├── subtask_1/                          # Binary Validity Classification
│   ├── symbolic_syllogism_engine.py    # Core engine: symbolic kernel + T5/LLM/rule-based parsers
│   ├── fuse_gemini_t5.py              # OR fusion of LLM + T5 predictions
│   ├── train_synthetic_parser.py       # Amnesiac T5 training on synthetic data
│   └── content_anonymizer.py           # Entity anonymization (ALPHA/BETA/GAMMA)
│
└── subtask_2/                          # Premise Retrieval + Validity
    ├── subtask2_engine.py              # Retrieval + symbolic validation pipeline
    ├── symbolic_syllogism_engine.py     # Symbolic engine (extended for retrieval)
    ├── z3_solver.py                    # Z3 theorem prover integration
    └── requirements.txt                # Dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r subtask_2/requirements.txt
```

<details>
<summary><b>Full dependency list</b></summary>

- `google-generativeai` — Gemini 2.0 Flash API
- `torch` — PyTorch for T5
- `transformers` — Hugging Face T5-Small
- `tqdm` — Progress bars
- `scikit-learn` — Evaluation metrics
- `spacy` — NLP utilities

</details>

### 2. Set API Key

```bash
export GEMINI_API_KEY='your-gemini-api-key'
```

### 3. Train the Amnesiac Parser (Subtask 1)

```bash
cd subtask_1
python train_synthetic_parser.py
```

This trains T5-Small on **synthetic nonsense syllogisms** only — the model never sees real-world entities during training.

**Training data example:**
```
Input:  "All ALPHA are BETA. No BETA is a GAMMA. Therefore, no ALPHA is a GAMMA."
Output: "Mood: AEE, Figure: 1"
```

### 4. Run Subtask 1 — Binary Validity

```bash
python symbolic_syllogism_engine.py
```

### 5. Run Subtask 2 — Retrieval + Validity

```bash
cd subtask_2
python subtask2_engine.py --test_data <path_to_test_data>
```

---

## 🔬 Key Design Decisions

### Why Amnesiac Training?

Standard fine-tuning on real syllogisms encodes correlations between entity types and validity — e.g., *"mammals"* → valid, *"unicorns"* → invalid. Training exclusively on ALPHA/BETA/GAMMA placeholders forces the model to learn **syntax only**.

### Why OR Fusion (not majority vote)?

| Strategy | Accuracy | TCE |
|----------|----------|-----|
| Gemini only | 91.43 | 9.52 |
| T5 only | 72.38 | 1.39 |
| **OR Fusion** | **97.38** | **3.10** |

OR fusion exploits complementary strengths: Gemini has high semantic recall; T5 has near-zero content effect. The OR gate recovers Gemini's false negatives while T5 suppresses content-driven false positives.

### Why not Z3 Theorem Proving?

Z3 is theoretically sound, but the NL→Z3 translation step re-introduces semantic content into the pipeline. Our experiments showed that Z3 *increased* content bias despite being a formal verifier. This is a significant negative result: **the introduction point of symbolic reasoning is as important as its use.**

---

## 📊 Valid Syllogistic Forms

The symbolic kernel checks against **15 valid forms** (with existential import):

| Figure 1 | Figure 2 | Figure 3 | Figure 4 |
|-----------|----------|----------|----------|
| AAA (Barbara) | EAE (Cesare) | AII (Datisi) | AEE (Camenes) |
| EAE (Celarent) | AEE (Camestres) | IAI (Disamis) | IAI (Dimaris) |
| AII (Darii) | EIO (Festino) | EIO (Ferison) | EIO (Fresison) |
| EIO (Ferio) | AOO (Baroco) | OAO (Bocardo) | |

---

## 📝 Citation

```bibtex
@inproceedings{lakksh-2026-semeval,
    title     = {Neuro-Symbolic Decomposition for Content-Bias-Free 
                 Syllogistic Reasoning},
    author    = {Lakksh Sharma, Krish Sharma, Jatin Bedi},
    booktitle = {Proceedings of the 20th International Workshop on 
                 Semantic Evaluation (SemEval-2026)},
    year      = {2026},
    publisher = {Association for Computational Linguistics}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Built for [SemEval-2026 Task 11](https://semeval.github.io/SemEval2026/) · Subtasks 1 & 2**

*Content bias is an architectural problem — solve it with architecture.*

</div>
