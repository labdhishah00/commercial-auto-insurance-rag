# PHASE: RERANKING EVALUATION

from reranking import retrieve_candidate, rerank_results

# 1. TEST QUESTIONS
# We use the same questions from earlier retrieval tests so that the comparison remains consistent.

test_queries = [
    "What is the policy period?",
    "What are hired autos?",
    "What are non-owned autos?",
    "Who is an insured under liability coverage?",
    "What is a temporary substitute auto?",
    "What does the Auto Medical Payments endorsement cover?",
    "What does covered auto symbol 7 mean?"
]

# 2. SETTINGS

SEMANTIC_K = 10
BM25_K = 10
FINAL_K = 4

# 3. LOOP THROUGH EACH QUERY

for query_number, query in enumerate(
    test_queries,
    start=1
):

    print("\n" + "#" * 100)

    print(
        f"QUERY {query_number}: {query}"
    )

    print("#" * 100)

    # 4. RETRIEVE HYBRID CANDIDATES
    #
    # retrieve_candidate() now combines:
    #
    # FAISS semantic search
    # +
    # BM25 keyword search

    retrieved_results = retrieve_candidate(
        query,
        semantic_k=SEMANTIC_K,
        bm25_k=BM25_K
    )

    # 5. RERANK HYBRID CANDIDATES


    reranked_results = rerank_results(
        query,
        retrieved_results,
        final_k=FINAL_K
    )

    # 6. BEFORE RERANKING

    print("\n" + "=" * 100)

    print(
        "BEFORE RERANKING - HYBRID CANDIDATES"
    )

    print("=" * 100)


    for index, item in enumerate(
        retrieved_results[:FINAL_K],
        start=1
    ):

        document = item["document"]

        faiss_score = item["faiss_score"]

        bm25_score = item["bm25_score"]

        source = item["source"]


        print("\n" + "-" * 80)

        print(
            f"HYBRID RESULT {index}"
        )


        print(
            "Retrieval source:",
            source
        )


        print(
            "FAISS distance:",
            faiss_score
        )


        print(
            "BM25 score:",
            bm25_score
        )


        print(
            "Form number:",
            document.metadata.get(
                "form_number"
            )
        )


        print(
            "Document type:",
            document.metadata.get(
                "document_type"
            )
        )


        print(
            "Parent document type:",
            document.metadata.get(
                "parent_document_type"
            )
        )


        print(
            "Section:",
            document.metadata.get(
                "section"
            )
        )


        print(
            "Page:",
            document.metadata.get(
                "page"
            )
        )


        print("\nTEXT:")

        print(
            document.page_content
        )

    # 7. AFTER RERANKING
  
    print("\n" + "=" * 100)

    print(
        "AFTER RERANKING - TOP 4"
    )

    print("=" * 100)


    for index, item in enumerate(
        reranked_results,
        start=1
    ):

        document = item["document"]


        print("\n" + "-" * 80)

        print(
            f"RERANKED RESULT {index}"
        )


        print(
            "Rerank score:",
            item["rerank_score"]
        )


        print(
            "Retrieval source:",
            item["retrieval_source"]
        )


        print(
            "Original FAISS distance:",
            item["faiss_score"]
        )


        print(
            "Original BM25 score:",
            item["bm25_score"]
        )


        print(
            "Form number:",
            document.metadata.get(
                "form_number"
            )
        )


        print(
            "Document type:",
            document.metadata.get(
                "document_type"
            )
        )


        print(
            "Parent document type:",
            document.metadata.get(
                "parent_document_type"
            )
        )


        print(
            "Section:",
            document.metadata.get(
                "section"
            )
        )


        print(
            "Page:",
            document.metadata.get(
                "page"
            )
        )


        print("\nTEXT:")

        print(
            document.page_content
        )