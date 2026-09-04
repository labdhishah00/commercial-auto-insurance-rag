# PHASE: HYBRID RETRIEVAL + RERANKING

from retrieval import semantic_search, bm25_search
from sentence_transformers import CrossEncoder

# 1. LOAD RERANKER MODEL

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# 2. BUILD HYBRID CANDIDATE SET

def retrieve_candidate(
    query,
    semantic_k=10,
    bm25_k=10
):
    """
    Retrieve candidates using BOTH:

    1. FAISS semantic search
    2. BM25 keyword search

    Then combine and deduplicate the results.

    Semantic retrieval helps with meaning-based questions.

    BM25 helps with exact terminology and identifiers.
    """

    # Get semantic candidates

    semantic_results = semantic_search(
        query,
        k=semantic_k
    )

    # Get BM25 candidates

    bm25_results = bm25_search(
        query,
        k=bm25_k
    )

    # Combine + deduplicate

    combined_results = []

    seen = set()

    # Add FAISS results first

    for document, score in semantic_results:

        key = (
            document.metadata.get("page"),
            document.page_content
        )


        if key not in seen:

            combined_results.append(
                {
                    "document": document,
                    "faiss_score": score,
                    "bm25_score": None,
                    "source": "semantic"
                }
            )

            seen.add(
                key
            )

    # Add BM25 results

    for document, score in bm25_results:

        key = (
            document.metadata.get("page"),
            document.page_content
        )


        if key not in seen:

            combined_results.append(
                {
                    "document": document,
                    "faiss_score": None,
                    "bm25_score": score,
                    "source": "bm25"
                }
            )

            seen.add(
                key
            )


    return combined_results

# 3. RERANK HYBRID CANDIDATES

def rerank_results(
    query,
    retrieved_results,
    final_k=4
):
    """
    Rerank combined FAISS + BM25 candidates.

    The CrossEncoder reads the query and document together
    and assigns a relevance score.

    Higher rerank score = more relevant.
    """

    # Handle empty retrieval results

    if not retrieved_results:
        return []

    # Build query-document pairs

    pairs = []


    for item in retrieved_results:

        document = item["document"]

        pairs.append(
            [
                query,
                document.page_content
            ]
        )

    # Score all candidates

    rerank_scores = reranker.predict(
        pairs
    )

    # Attach reranking scores

    reranked_results = []


    for index, item in enumerate(
        retrieved_results
    ):

        reranked_results.append(
            {
                "document": item["document"],

                "faiss_score": item[
                    "faiss_score"
                ],

                "bm25_score": item[
                    "bm25_score"
                ],

                "retrieval_source": item[
                    "source"
                ],

                "rerank_score": float(
                    rerank_scores[index]
                )
            }
        )
        
    # Sort by CrossEncoder relevance
    # Higher score = more relevant.

    reranked_results = sorted(
        reranked_results,
        key=lambda item: item[
            "rerank_score"
        ],
        reverse=True
    )


    return reranked_results[:final_k]