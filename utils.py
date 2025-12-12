# utils.py
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate



model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def cos_sim(a, b):
    return (a @ b.T) / (a.norm() * b.norm())

def similarity(a, b):
    # This relies on 'model' being loaded in main as a global variable
    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)
    return float(cos_sim(emb_a, emb_b))


SHORT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are an EXTRACTIVE question-answering model used for reinforcement learning.

You MUST answer ONLY using the information found in the provided CONTEXT.

====================================
### HOW YOU MUST ANSWER
- Respond with **one short phrase** OR **one short sentence**.
- Prefer **verbatim words** from the context whenever possible.
- DO NOT add explanations.
- DO NOT rewrite or expand the answer.
- DO NOT use any outside knowledge.
- If the answer is NOT explicitly present in the context, respond EXACTLY:
  **"I don't know."**

====================================
### GOOD ANSWERS (correct behavior)
Context: "Napoleon invaded Russia in 1812."
Q: "When did Napoleon invade Russia?"
A: "1812."          ← Correct, short, from context

Context: "The mitochondria is the powerhouse of the cell."
Q: "What is the powerhouse of the cell?"
A: "The mitochondria."   ← Exact wording

Context: "The treaty was signed by Spain and Portugal."
Q: "Who signed the treaty?"
A: "Spain and Portugal."  ← Extracted list

Context: "This text contains no dates."
Q: "When was the event?"
A: "I don't know."   ← Because nothing is stated

====================================
### BAD ANSWERS (never do this)
A: "Napoleon invaded Russia in 1812 during his failed campaign."
    (✘ Too long, adds external info)

A: "Probably sometime in the 19th century."
    (✘ Guess, not allowed)

A: "Spain, Portugal, and France."
    (✘ Adds facts not in context)

A: "The mitochondria, an organelle responsible for energy."
    (✘ Adds explanation)

====================================

### NOW ANSWER STRICTLY ACCORDING TO THESE RULES.

CONTEXT:
{context}

QUESTION:
{question}

SHORT ANSWER:
""")
