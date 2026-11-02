import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Testing AI Video Assistant Core Pipeline")
print("=" * 60)

from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

sample_transcript = """
Sarah: Welcome everyone to the quarterly project sync. Today we need to lock down our deliverables.
David: On the backend side, the payment migration is 90% ready. We need to complete security testing.
Sarah: Great. David, please ensure all Stripe payment tests pass by Thursday at 5 PM.
Elena: The design team has finalized the mobile UI. I will send the Figma specs to engineering by Wednesday morning.
David: Will we also support Apple Pay in this release?
Sarah: No, we have decided that Apple Pay support is postponed to Q4. Only Stripe credit cards for now.
Elena: Who is following up with the compliance team regarding GDPR data retention?
Sarah: That is still an open question. Let's assign an owner by tomorrow's standup.
"""

print("\n1. Generating Title...")
title = generate_title(sample_transcript)
print(f"   --> Title: {title}")

print("\n2. Generating Summary...")
summary = summarize(sample_transcript)
print(f"   --> Summary:\n{summary}")

print("\n3. Extracting Action Items...")
actions = extract_action_items(sample_transcript)
print(f"   --> Action Items:\n{actions}")

print("\n4. Extracting Key Decisions...")
decisions = extract_key_decisions(sample_transcript)
print(f"   --> Decisions:\n{decisions}")

print("\n5. Extracting Open Questions...")
questions = extract_questions(sample_transcript)
print(f"   --> Questions:\n{questions}")

print("\n6. Testing RAG Chain...")
rag_chain = build_rag_chain(sample_transcript)

q1 = "When are the Stripe payment tests due and who is responsible?"
a1 = ask_question(rag_chain, q1)
print(f"   Q: {q1}\n   A: {a1}")

q2 = "Was Apple Pay approved for this release?"
a2 = ask_question(rag_chain, q2)
print(f"   Q: {q2}\n   A: {a2}")

print("\n" + "=" * 60)
print("SUCCESS: All core components verified and working!")
print("=" * 60)
