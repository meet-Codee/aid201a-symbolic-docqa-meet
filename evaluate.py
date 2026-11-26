#!/usr/bin/env python3
"""
Automated 18-Query Benchmark Evaluation Harness (CO-4)
Course: AID201A Artificial Intelligence
Student: Meet Amit Dharaiya (Reg. No.: 25ETAI410081)
"""

from pathlib import Path
from docqa import ABSTAIN_MSG, answer_query, load_document

BENCHMARK = [
    ("Q01", "Which building houses the AI and Robotics Lab?", "Tech Park Block B", "Direct"),
    ("Q02", "Who is the course leader for AID201A Artificial Intelligence?", "Dr. P. Padma Priya Dharishini", "Direct"),
    ("Q03", "What is the workstation capacity of AI and Robotics Lab?", "60 Workstations", "Direct"),
    ("Q04", "What is the power rating of High Performance Computing Lab?", "45 kVA Redundant UPS", "Direct"),
    ("Q05", "Which department offers AID201A Artificial Intelligence?", "Department of CSE", "Direct"),
    ("Q06", "How many credits is AID201A Artificial Intelligence?", "4 Credits (3rd Semester)", "Direct"),
    ("Q07", "Where is the NVIDIA DGX A100 Cluster located?", "Tech Park Block B; Peenya Campus; Bengaluru North", "Forward (R1+R3)"),
    ("Q08", "Where is the TurtleBot4 Autonomous Rover located?", "Tech Park Block B; Peenya Campus; Bengaluru North", "Forward (R1+R3)"),
    ("Q09", "Where is the Embedded Systems Lab located?", "Peenya Campus; Bengaluru North", "Forward (R2+R3)"),
    ("Q10", "What are all prerequisites for AID201A Artificial Intelligence?", "CSD102A Data Structures; CSD101A Python Programming", "Forward (R6)"),
    ("Q11", "Where is the FPGA Logic Analyzer Rig located?", "Engineering Annexe; Peenya Campus; Bengaluru North", "Forward (R1+R3)"),
    ("Q12", "Which department manages the NVIDIA DGX A100 Cluster?", "Department of CSE", "Backward (R4)"),
    ("Q13", "Which faculty oversees the AI and Robotics Lab?", "Faculty of Engineering and Technology", "Backward (R5)"),
    ("Q14", "How is TurtleBot4 Autonomous Rover connected to Dr. P. Padma Priya Dharishini?", "TurtleBot4 Autonomous Rover --[installed_in]--> AI and Robotics Lab <--[uses_facility]-- AID201A Artificial Intelligence --[led_by]--> Dr. P. Padma Priya Dharishini", "BFS Graph Search"),
    ("Q15", "What is the path between FPGA Logic Analyzer Rig and Faculty of Engineering and Technology?", "FPGA Logic Analyzer Rig --[installed_in]--> Embedded Systems Lab --[managed_by]--> Department of ECE --[part_of_faculty]--> Faculty of Engineering and Technology", "BFS Graph Search"),
    ("Q16", "What is the tuition fee for AID201A Artificial Intelligence?", ABSTAIN_MSG, "Abstention"),
    ("Q17", "Where is the Quantum Photonics Lab located?", ABSTAIN_MSG, "Abstention"),
    ("Q18", "What is the power rating of Embedded Systems Lab?", ABSTAIN_MSG, "Abstention"),
]


def run_evaluation() -> None:
    kb_path = Path(__file__).parent / "campus_kb.csv"
    kb = load_document(kb_path)
    passed = 0
    answered = 0
    abstained = 0
    correct_abstentions = 0

    print(f"Loaded KB: {len(kb.explicit_facts)} explicit facts, {len(kb.closure_facts)} total closure facts.\n")
    for qid, question, expected, qtype in BENCHMARK:
        res = answer_query(kb, question)
        actual = str(res["answer"])
        ok = actual == expected
        if ok:
            passed += 1
        if res["status"] == "ANSWERED":
            answered += 1
        else:
            abstained += 1
            if expected == ABSTAIN_MSG:
                correct_abstentions += 1
        status_tag = "PASS" if ok else "FAIL"
        print(f"[{status_tag}] {qid} ({qtype}): {question}")
        print(f"       Output:     {actual}")
        print(f"       Provenance: {res.get('provenance')}\n")

    total = len(BENCHMARK)
    print("=" * 78)
    print(f"Exact-Answer Accuracy: {passed}/{total} = {passed / total * 100:.2f}%")
    print(f"Answer Coverage:       {answered}/{total} = {answered / total * 100:.2f}%")
    print(f"Abstention Precision:  {correct_abstentions}/{abstained} = {correct_abstentions / max(1, abstained) * 100:.2f}%")
    print("=" * 78)
    assert passed == total, f"Expected {total}/{total} passed, got {passed}/{total}"


if __name__ == "__main__":
    run_evaluation()
