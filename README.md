# Symbolic Document Question Answering System (`AID201A`)

**Student Name:** Meet Amit Dharaiya  
**Register Number:** `25ETAI410081`  
**Course:** `AID201A` — Artificial Intelligence (3rd Semester B.Tech. CSE, 2025 Batch)  
**Institution:** Faculty of Engineering & Technology, Ramaiah University of Applied Sciences (RUAS)

---

## 1. Overview & Constraint Compliance

This repository contains the complete Python implementation, sample domain knowledge source, automated evaluation benchmark suite, and 6-page IEEE technical report for **Part A** of Assignment 1:
- **Zero NLP / Zero RAG / Zero LLM**: All document ingestion, knowledge representation, inference, graph search, and answer synthesis are 100% deterministic and symbolic.
- **Knowledge Representation**: Typed Subject–Relation–Object–Provenance (SROP) Fact Frames indexed by $O(1)$ hash maps ($I_{SR}, I_R$) and an undirected Semantic Adjacency Graph $G = (V, E)$.
- **Reasoning & Search**:
  1. **Semi-Naive Forward-Chaining Deductive Closure** over 6 First-Order Horn Clauses (`R1`–`R6`) covering equipment-to-building containment, facility-to-campus spatial bridges, transitive spatial inclusion, custodial inheritance, organizational hierarchy propagation, and prerequisite chain transitivity.
  2. **Goal-Directed Backward Proof Trace** recording exact rule and row provenance for every derived conclusion.
  3. **Breadth-First Search (BFS)** over the Semantic Graph to discover the shortest relational path connecting arbitrary domain entities.
  4. **Closed-World Assumption (CWA)** deterministic abstention on unsupported queries (0% hallucination).

---

## 2. Files Included

| File | Description |
|---|---|
| `AID201A_Meet_Amit_Dharaiya_Assignment.docx` | Complete 6-page word-processed assignment submission (Page 1 preserved RUAS cover sheet + Pages 2–6 full-width IEEE technical report) |
| `docqa.py` | Core Symbolic DQA Agent (multi-format `.csv`/`.json`/`.txt` document parser, Horn-clause inference engine, BFS path finder, CLI interface) |
| `evaluate.py` | Automated 18-query benchmark evaluation script computing accuracy, coverage, and abstention precision |
| `campus_kb.csv` | RUAS Academic, Laboratory, Equipment & Course Register (22 explicit domain facts expanding to 49 closure facts) |

---

## 3. How to Run

```bash
# 1. Run a single-shot deductive multi-hop question
python docqa.py campus_kb.csv --question "Where is the NVIDIA DGX A100 Cluster located?"

# 2. Run a BFS relational graph path search query
python docqa.py campus_kb.csv --question "How is TurtleBot4 Autonomous Rover connected to Dr. P. Padma Priya Dharishini?"

# 3. Run the full 18-question automated benchmark evaluation (CO-4)
python evaluate.py
```
