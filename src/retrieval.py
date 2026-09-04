# PHASE: RETRIEVAL - SEMANTIC SEARCH + BM25 KEYWORD SEARCH

from vector_store import vector_store
from chunking import final_chunks
from rank_bm25 import BM25Okapi

import re

# 1. TOKENIZATION

def tokenize(text):
    """
    Convert text into simple lowercase word/number tokens.

    Example:

        "What does Symbol 7 mean?"

    becomes:

        [
            "what",
            "does",
            "symbol",
            "7",
            "mean"
        ]

    BM25 works with tokens rather than embeddings.
    """

    text = text.lower()

    tokens = re.findall(
        r"\b\w+\b",
        text
    )

    return tokens

# 2. BUILD BM25 CORPUS

# final_chunks contains LangChain Document objects.
#
# BM25 needs tokenized text rather than Document objects.

bm25_corpus = []


for chunk in final_chunks:

    text = chunk.page_content

    tokens = tokenize(
        text
    )

    bm25_corpus.append(
        tokens
    )


# Create the BM25 search index.
bm25 = BM25Okapi(
    bm25_corpus
)

# 3. SEMANTIC SEARCH

def semantic_search(query, k=4):
    """
    Retrieve top-k chunks using FAISS semantic similarity.

    Input:
        query -> user question
        k     -> number of results

    Output:
        list of:

        (
            Document,
            FAISS distance score
        )

    Lower FAISS distance generally indicates a closer
    semantic match.
    """

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    return results

# 4. BM25 KEYWORD SEARCH

def bm25_search(query, k=4):
    """
    Retrieve top-k chunks using BM25 keyword matching.

    Input:
        query -> user question
        k     -> number of results

    Output:
        list of:

        (
            Document,
            BM25 score
        )

    Higher BM25 score indicates a stronger lexical match.
    """

    # Convert the user's question into tokens.
    query_tokens = tokenize(
        query
    )


    # Calculate a BM25 score for every policy chunk.
    scores = bm25.get_scores(
        query_tokens
    )


    # Sort chunk indexes from highest BM25 score
    # to lowest BM25 score.
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )


    # Keep only the strongest k indexes.
    top_indexes = ranked_indexes[:k]


    results = []


    for index in top_indexes:

        # Original LangChain Document.
        chunk = final_chunks[index]

        # BM25 score for this document.
        score = scores[index]


        results.append(
            (
                chunk,
                score
            )
        )


    return results

# 5. OPTIONAL RETRIEVAL DEMONSTRATION
#
# This section runs ONLY when we execute:
#
#     python retrieval.py
#
# It does NOT run when another module imports:
#
#     semantic_search
#     bm25_search
#
# This prevents unwanted debug output in rag_evaluation.py.

if __name__ == "__main__":

    query = (
        "What does covered auto symbol 7 mean?"
    )


    semantic_results = semantic_search(
        query,
        k=4
    )


    bm25_results = bm25_search(
        query,
        k=4
    )

    # PRINT QUERY

    print(
        "\n" + "#" * 100
    )

    print(
        "QUERY:"
    )

    print(
        query
    )

    print(
        "#" * 100
    )

    # PRINT SEMANTIC SEARCH RESULTS

    print(
        "\n" + "#" * 100
    )

    print(
        "SEMANTIC SEARCH RESULTS - FAISS"
    )

    print(
        "#" * 100
    )


    for index, (result, score) in enumerate(
        semantic_results,
        start=1
    ):

        print(
            "\n" + "=" * 80
        )

        print(
            f"SEMANTIC RESULT {index}"
        )

        print(
            "=" * 80
        )


        print(
            "FAISS distance:",
            score
        )


        print(
            "Form number:",
            result.metadata.get(
                "form_number"
            )
        )


        print(
            "Document type:",
            result.metadata.get(
                "document_type"
            )
        )


        print(
            "Parent document type:",
            result.metadata.get(
                "parent_document_type"
            )
        )


        print(
            "Section:",
            result.metadata.get(
                "section"
            )
        )


        print(
            "Page:",
            result.metadata.get(
                "page"
            )
        )


        print(
            "\nRETRIEVED TEXT:"
        )

        print(
            result.page_content
        )

    # PRINT BM25 SEARCH RESULTS

    print(
        "\n" + "#" * 100
    )

    print(
        "BM25 KEYWORD SEARCH RESULTS"
    )

    print(
        "#" * 100
    )


    for index, (result, score) in enumerate(
        bm25_results,
        start=1
    ):

        print(
            "\n" + "=" * 80
        )

        print(
            f"BM25 RESULT {index}"
        )

        print(
            "=" * 80
        )


        print(
            "BM25 score:",
            score
        )


        print(
            "Form number:",
            result.metadata.get(
                "form_number"
            )
        )


        print(
            "Document type:",
            result.metadata.get(
                "document_type"
            )
        )


        print(
            "Parent document type:",
            result.metadata.get(
                "parent_document_type"
            )
        )


        print(
            "Section:",
            result.metadata.get(
                "section"
            )
        )


        print(
            "Page:",
            result.metadata.get(
                "page"
            )
        )


        print(
            "\nRETRIEVED TEXT:"
        )

        print(
            result.page_content
        )