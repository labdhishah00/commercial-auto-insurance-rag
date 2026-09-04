# Phase: Chunking strategy retrieval comparison

from vector_store import vector_store
from vector_store_strategy2 import vector_store_strategy2

test_queries = [
    "What is the policy period?",
    "What are hired autos?",
    "What are non-owned autos?",
    "Who is an insured under liability coverage?",
    "What is a temporary substitute auto?",
    "What does the Auto Medical Payments endorsement cover?",
    "What does covered auto symbol 7 mean?"
]

TOP_K = 4

for query_number, query in enumerate(test_queries, start=1):
    print("\n" + "#" * 100)
    print(f"QUERY {query_number} : {query}")
    print("#" * 100)

    strategy1_results = (vector_store.similarity_search_with_score(query, k = TOP_K))

    strategy2_results = (vector_store_strategy2.similarity_search_with_score(query, k = TOP_K))

    print("\n" + "-" * 100)
    print("strategy 1 results")
    print("Recursive chunking: 1000chars / 150 overlap")
    print("-" * 100)

    for result_index, (result, score) in enumerate(strategy1_results,start=1):

        print("\n" + "=" * 80)

        print(f"STRATEGY 1 - RESULT {result_index}")

        print("=" * 80)

        print("FAISS distance:",score)


        # Insurance metadata
        print("Form number:",result.metadata.get("form_number"))


        print("Document type:",result.metadata.get("document_type"))

        print("Parent document type:",result.metadata.get("parent_document_type"))


        print("Section:", result.metadata.get("section"))

        print("Page:", result.metadata.get("page"))

        print("Chunk heading:", result.metadata.get("chunk_heading"))

        # Retrieved text
        print("\nRETRIEVED TEXT:")

        print(result.page_content)

# PRINT STRATEGY 2 RESULTS
   

    print("\n" + "-" * 100)

    print("STRATEGY 2 RESULTS")

    print("Structure-aware / section-aware chunking")

    print("-" * 100)


    for result_index, (result, score) in enumerate(strategy2_results,start=1):

        print("\n" + "=" * 80)

        print(f"STRATEGY 2 - RESULT {result_index}")

        print("=" * 80)

        # FAISS distance
        print("FAISS distance:",score)

        # Insurance metadata
        print("Form number:",result.metadata.get("form_number"))
        print("Document type:",result.metadata.get("document_type"))
        print("Parent document type:", result.metadata.get("parent_document_type"))
        print("Section:",result.metadata.get("section"))
        print("Page:",result.metadata.get("page"))

        # This metadata mainly exists in Strategy 2.
        print("Chunk heading:",result.metadata.get("chunk_heading"))
        # Retrieved text
        print("\nRETRIEVED TEXT:")
        print(result.page_content)