# PHASE: FINAL RAG EVALUATION

from rag import answer_question

# EVALUATION QUESTIONS  We intentionally reuse questions from earlier retrieval experiments so that our final evaluation remains consistent.

test_queries = [

    # CATEGORY 1: DIRECT POLICY QUESTIONS

    "What is the policy period?",

    "What are hired autos?",

    "What are non-owned autos?",

    "Who is an insured under liability coverage?",

    "What is a temporary substitute auto?",

    # CATEGORY 2: COVERAGE / ENDORSEMENT QUESTIONS

    "What does the Auto Medical Payments endorsement cover?",

    "Who is an insured under the Auto Medical Payments endorsement?",

    "What is the limit of insurance for Auto Medical Payments Coverage?",

    # CATEGORY 3: EXACT TERM / IDENTIFIER QUESTIONS

    "What does covered auto symbol 7 mean?",

    "What does covered auto symbol 1 mean?",

    # CATEGORY 4: MORE COMPLEX POLICY QUESTIONS

    "When can a non-owned auto be used as a temporary substitute for a covered auto?",

    "Are medical payments available for someone occupying a covered auto?",

    # CATEGORY 5: AMBIGUOUS / COVERAGE QUESTIONS

    "Are rented trucks covered?",

    "Is every vehicle owned by the insured automatically covered?",

    # CATEGORY 6: INSUFFICIENT-EVIDENCE TEST

    "Was vehicle VIN 1ABC23456789 covered on January 15, 2024?"
]

# RUN FINAL RAG EVALUATION

for query_number, query in enumerate(test_queries, start=1):

    print("\n" + "#" * 100)

    print(f"QUESTION {query_number}:")

    print(query)

    print("#" * 100)

    # Run the complete RAG pipeline:
    #
    # FAISS
    #   ↓
    # reranking
    #   ↓
    # evidence expansion
    #   ↓
    # corrupted text filtering
    #   ↓
    # LLM answer

    answer = answer_question(query)

    print("\nRAG ANSWER:")

    print(answer)