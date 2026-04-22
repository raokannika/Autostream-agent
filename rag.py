import json
import os
from typing import List


def load_knowledge_base(path: str = "knowledge_base.json") -> dict:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, path)
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents(kb: dict) -> List[str]:
    docs = []

    company = kb.get("company", {})
    docs.append(
        f"Company: {company.get('name')}. "
        f"Tagline: {company.get('tagline')}. "
        f"Description: {company.get('description')}"
    )

    for plan in kb.get("pricing", []):
        features = "; ".join(plan.get("features", []))
        limitations = "; ".join(plan.get("limitations", [])) or "None"
        docs.append(
            f"Plan: {plan['plan']}. "
            f"Price: {plan['price_label']}. "
            f"Features: {features}. "
            f"Limitations: {limitations}."
        )

    for policy in kb.get("policies", []):
        docs.append(
            f"Policy – {policy['topic']}: {policy['detail']}"
        )

    for faq in kb.get("faqs", []):
        docs.append(
            f"FAQ – Q: {faq['question']} A: {faq['answer']}"
        )

    return docs


def retrieve(query: str, top_k: int = 3) -> str:
    kb = load_knowledge_base()
    docs = build_documents(kb)
    query_lower = query.lower()
    scored = []
    for doc in docs:
        doc_lower = doc.lower()
        score = sum(1 for word in query_lower.split() if word in doc_lower)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:top_k]]
    return "\n\n".join(top_docs) if top_docs else "No relevant information found."
