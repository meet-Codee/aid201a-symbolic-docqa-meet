#!/usr/bin/env python3
"""
Symbolic Document Question Answering System (AID201A - Artificial Intelligence)
Student: Meet Amit Dharaiya (Reg. No.: 25ETAI410081)
Ramaiah University of Applied Sciences (RUAS)

Constraint Compliance:
- Purely symbolic Knowledge Representation (Typed SROP Frames + Semantic Graph)
- First-Order Horn-Clause Forward-Chaining Deductive Closure & Backward Proof Trace
- Uninformed Breadth-First Search (BFS) for Multi-Hop Relational Path Discovery
- Zero NLP, Zero Embeddings, Zero LLMs, Zero RAG, Python 3.10+ Standard Library Only
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ABSTAIN_MSG = "ABSTAIN: No verifiable fact or deductive proof exists in the Knowledge Base for this query."

# Canonical Ontology Relations and Deterministic Surface Cue Lexicon
RELATION_CUES: Dict[str, Tuple[str, ...]] = {
    "housed_in": ("which building houses", "housed in", "housed", "building houses"),
    "located_in": ("where is", "located", "location", "which campus", "where"),
    "installed_in": ("installed in", "which lab has", "installed"),
    "managed_by": ("which department manages", "managed by", "manages", "custodian", "managed"),
    "under_faculty": ("which faculty oversees", "oversees", "under faculty", "faculty oversees", "deanery"),
    "part_of_faculty": ("part of faculty", "which faculty is"),
    "offered_by": ("which department offers", "offered by", "offers"),
    "led_by": ("course leader", "led by", "who leads", "who teaches", "instructor"),
    "requires_prereq": ("prerequisites", "prerequisite", "prereq", "required before"),
    "uses_facility": ("uses facility", "which lab is used by", "practical lab"),
    "has_capacity": ("workstation capacity", "capacity", "how many workstations"),
    "power_rating": ("power rating", "ups", "kva", "power"),
    "max_credits": ("how many credits", "credits", "credit weight"),
}

PATH_CUES: Tuple[str, ...] = ("connected to", "path between", "relationship between", "how is", "linked to")

# First-Order Horn-Clause Rule Base: (rule_id, head_rel, body_rel_1, body_rel_2, description)
HORN_RULES: List[Tuple[str, str, str, str, str]] = [
    ("R1", "located_in", "installed_in", "housed_in", "installed_in(X,Y) ^ housed_in(Y,Z) => located_in(X,Z)"),
    ("R2", "located_in", "housed_in", "located_in", "housed_in(X,Y) ^ located_in(Y,Z) => located_in(X,Z)"),
    ("R3", "located_in", "located_in", "located_in", "located_in(X,Y) ^ located_in(Y,Z) => located_in(X,Z)"),
    ("R4", "managed_by", "installed_in", "managed_by", "installed_in(X,Y) ^ managed_by(Y,D) => managed_by(X,D)"),
    ("R5", "under_faculty", "managed_by", "part_of_faculty", "managed_by(X,D) ^ part_of_faculty(D,F) => under_faculty(X,F)"),
    ("R6", "requires_prereq", "requires_prereq", "requires_prereq", "requires_prereq(C1,C2) ^ requires_prereq(C2,C3) => requires_prereq(C1,C3)"),
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


@dataclass(frozen=True)
class Fact:
    fact_id: str
    subject: str
    relation: str
    obj: str
    source: str
    is_inferred: bool = False
    depth: int = 1


@dataclass
class QueryFrame:
    raw_question: str
    normalized: str
    mode: str  # "DEDUCTIVE_LOOKUP" or "PATH_SEARCH"
    relation: Optional[str]
    entities: List[str] = field(default_factory=list)


class KnowledgeBase:
    def __init__(self) -> None:
        self.explicit_facts: List[Fact] = []
        self.closure_facts: List[Fact] = []
        self.by_sr: Dict[Tuple[str, str], List[Fact]] = {}
        self.by_rel: Dict[str, List[Fact]] = {}
        self.entities: Dict[str, str] = {}  # norm_name -> display_name
        self.graph: Dict[str, List[Tuple[str, str, str]]] = {}  # norm_u -> [(edge_label, norm_v, fact_id)]

    def add_fact(self, fact: Fact) -> bool:
        s_k, r_k, o_k = normalize_text(fact.subject), fact.relation.strip().lower(), normalize_text(fact.obj)
        existing = self.by_sr.get((s_k, r_k), [])
        if any(normalize_text(f.obj) == o_k for f in existing):
            return False
        if fact.is_inferred:
            self.closure_facts.append(fact)
        else:
            self.explicit_facts.append(fact)
            self.closure_facts.append(fact)
        self.by_sr.setdefault((s_k, r_k), []).append(fact)
        self.by_rel.setdefault(r_k, []).append(fact)
        self.entities[s_k] = fact.subject
        self.entities[o_k] = fact.obj
        self.graph.setdefault(s_k, []).append((f"--[{r_k}]-->", o_k, fact.fact_id))
        self.graph.setdefault(o_k, []).append((f"<--[{r_k}]--", s_k, fact.fact_id))
        return True

    def compute_forward_closure(self) -> int:
        derived_count = 0
        added = True
        while added:
            added = False
            for rule_id, r_head, r_b1, r_b2, _ in HORN_RULES:
                b1_facts = list(self.by_rel.get(r_b1, []))
                for f1 in b1_facts:
                    mid_key = normalize_text(f1.obj)
                    b2_facts = list(self.by_sr.get((mid_key, r_b2), []))
                    for f2 in b2_facts:
                        derived_count_candidate = len(self.closure_facts) - len(self.explicit_facts) + 1
                        fid = f"D{derived_count_candidate:02d}"
                        trace = f"{rule_id}({f1.fact_id} [{f1.source}] + {f2.fact_id} [{f2.source}])"
                        new_fact = Fact(
                            fact_id=fid,
                            subject=f1.subject,
                            relation=r_head,
                            obj=f2.obj,
                            source=trace,
                            is_inferred=True,
                            depth=f1.depth + f2.depth,
                        )
                        if self.add_fact(new_fact):
                            derived_count += 1
                            added = True
        return derived_count

    def query_subject_relation(self, subject: str, relation: str) -> List[Fact]:
        matches = list(self.by_sr.get((normalize_text(subject), relation), []))
        matches.sort(key=lambda f: (f.depth, f.fact_id))
        return matches


def load_document(path: Path) -> KnowledgeBase:
    kb = KnowledgeBase()
    suffix = path.suffix.lower()
    raw_rows: List[Tuple[str, str, str, str]] = []

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"subject", "relation", "object"}
            if not reader.fieldnames or not required.issubset({h.strip().lower() for h in reader.fieldnames}):
                raise ValueError(f"CSV file {path.name} must include header columns: subject, relation, object")
            for idx, row in enumerate(reader, start=1):
                s = (row.get("subject") or "").strip()
                r = (row.get("relation") or "").strip().lower()
                o = (row.get("object") or "").strip()
                if not s or not r or not o:
                    raise ValueError(f"Malformed record at {path.name}:row {idx}")
                raw_rows.append((s, r, o, f"{path.name}:row {idx}"))
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload["facts"] if isinstance(payload, dict) and "facts" in payload else payload
        for idx, item in enumerate(items, start=1):
            s = str(item.get("subject", "")).strip()
            r = str(item.get("relation", "")).strip().lower()
            o = str(item.get("object", "")).strip()
            if not s or not r or not o:
                raise ValueError(f"Malformed JSON object at {path.name}:item {idx}")
            raw_rows.append((s, r, o, f"{path.name}:item {idx}"))
    else:
        # Stanza / line-based structured technical text parser (.txt / .md)
        current_entity = None
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                current_entity = stripped[1:-1].replace("Entity:", "").strip()
            elif "|" in stripped:
                parts = [p.strip() for p in stripped.split("|")]
                if len(parts) == 3:
                    raw_rows.append((parts[0], parts[1].lower(), parts[2], f"{path.name}:line {idx}"))
            elif ":" in stripped and current_entity:
                k, v = [p.strip() for p in stripped.split(":", 1)]
                raw_rows.append((current_entity, k.lower().replace(" ", "_"), v, f"{path.name}:line {idx}"))

    for idx, (s, r, o, src) in enumerate(raw_rows, start=1):
        kb.add_fact(Fact(fact_id=f"F{idx:02d}", subject=s, relation=r, obj=o, source=src, is_inferred=False, depth=1))
    kb.compute_forward_closure()
    return kb


def compile_query_frame(kb: KnowledgeBase, question: str) -> QueryFrame:
    norm_q = normalize_text(question)

    # Identify all entity mentions sorted by longest span and position in question
    found_entities: List[Tuple[int, int, str]] = []
    padded_q = f" {norm_q} "
    for norm_ent, disp_ent in sorted(kb.entities.items(), key=lambda x: len(x[0]), reverse=True):
        pos = padded_q.find(f" {norm_ent} ")
        if pos != -1:
            # Check overlap with already matched longer entity spans
            span_start, span_end = pos, pos + len(norm_ent)
            if not any(not (span_end <= s or span_start >= e) for s, e, _ in found_entities):
                found_entities.append((span_start, span_end, disp_ent))
    found_entities.sort(key=lambda x: x[0])
    entities = [e[2] for e in found_entities]

    # Check if query requests relational graph path between two entities
    if len(entities) >= 2 and any(cue in norm_q for cue in PATH_CUES):
        return QueryFrame(raw_question=question, normalized=norm_q, mode="PATH_SEARCH", relation="path", entities=entities[:2])

    # Score canonical relations by longest matching cue phrase
    best_rel: Optional[str] = None
    best_score = 0
    for rel, cues in RELATION_CUES.items():
        for cue in cues:
            norm_cue = normalize_text(cue)
            if f" {norm_cue} " in padded_q:
                score = len(norm_cue.split()) * 10 + len(norm_cue)
                if score > best_score:
                    best_score = score
                    best_rel = rel

    return QueryFrame(raw_question=question, normalized=norm_q, mode="DEDUCTIVE_LOOKUP", relation=best_rel, entities=entities)


def bfs_relational_path(kb: KnowledgeBase, start_entity: str, goal_entity: str) -> Dict[str, object]:
    s_k, g_k = normalize_text(start_entity), normalize_text(goal_entity)
    if s_k not in kb.graph or g_k not in kb.graph:
        return {"status": "ABSTAIN", "answer": ABSTAIN_MSG, "reason": "One or both entities absent from KB graph."}

    queue: deque[Tuple[str, List[str], List[str]]] = deque([(s_k, [kb.entities[s_k]], [])])
    visited: Set[str] = {s_k}

    while queue:
        curr, chain_str, fact_ids = queue.popleft()
        if curr == g_k:
            return {
                "status": "ANSWERED",
                "mode": "BFS_GRAPH_SEARCH",
                "answer": " ".join(chain_str),
                "hops": len(fact_ids),
                "provenance": f"BFS shortest path ({len(fact_ids)} edges): " + " -> ".join(fact_ids),
            }
        for edge_label, neighbor, fid in kb.graph.get(curr, []):
            if fid.startswith("D"):
                continue  # Traverse base explicit semantic edges for minimal physical path
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, chain_str + [edge_label, kb.entities[neighbor]], fact_ids + [fid]))

    return {"status": "ABSTAIN", "answer": ABSTAIN_MSG, "reason": "No connected path found in Semantic Graph."}


def answer_query(kb: KnowledgeBase, question: str) -> Dict[str, object]:
    t0 = time.perf_counter()
    qf = compile_query_frame(kb, question)

    if qf.mode == "PATH_SEARCH" and len(qf.entities) >= 2:
        res = bfs_relational_path(kb, qf.entities[0], qf.entities[1])
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return res

    if not qf.relation or not qf.entities:
        return {
            "status": "ABSTAIN",
            "mode": "CWA_ABSTENTION",
            "answer": ABSTAIN_MSG,
            "provenance": "Unrecognized entity or unmapped relation cue under Closed-World Assumption.",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    matches = kb.query_subject_relation(qf.entities[0], qf.relation)
    if not matches:
        return {
            "status": "ABSTAIN",
            "mode": "CWA_ABSTENTION",
            "answer": ABSTAIN_MSG,
            "provenance": f"No explicit or derivable fact for ({qf.entities[0]}, {qf.relation}, ?).",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }

    ans_values = "; ".join(f.obj for f in matches)
    prov_values = "; ".join(f"{f.fact_id}: {f.source}" for f in matches)
    mode = "INFERRED_HORN_CLOSURE" if any(f.is_inferred for f in matches) else "DIRECT_KB_LOOKUP"
    return {
        "status": "ANSWERED",
        "mode": mode,
        "subject": qf.entities[0],
        "relation": qf.relation,
        "answer": ans_values,
        "provenance": prov_values,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Symbolic Document Question Answering Agent (AID201A)")
    parser.add_argument("document", type=Path, help="Path to input knowledge document (.csv, .json, .txt)")
    parser.add_argument("--question", "-q", type=str, help="Question to answer from the Knowledge Base")
    args = parser.parse_args()

    t0 = time.perf_counter()
    kb = load_document(args.document)
    ingest_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    n_exp = len(kb.explicit_facts)
    n_inf = len(kb.closure_facts) - n_exp
    print(f"[KB STATUS] Ingested {n_exp} explicit facts | Derived {n_inf} inferred facts | Total Closure = {len(kb.closure_facts)} facts ({ingest_ms} ms)")

    if args.question:
        res = answer_query(kb, args.question)
        print(f"[MODE]       {res.get('mode')}")
        print(f"[ANSWER]     {res['answer']}")
        print(f"[PROVENANCE] {res.get('provenance')}")
        print(f"[LATENCY]    {res.get('latency_ms')} ms")
    else:
        print("Interactive mode started. Type a question or 'exit' to quit.")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except EOFError:
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            res = answer_query(kb, q)
            print(f"  Answer:     {res['answer']}")
            print(f"  Provenance: {res.get('provenance')} ({res.get('latency_ms')} ms)")


if __name__ == "__main__":
    main()
