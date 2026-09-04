# PHASE: BASIC RAG ANSWER GENERATION

from reranking import (retrieve_candidate,rerank_results, reranker)
from chunking import final_chunks
from langchain_ollama import ChatOllama

# 1. LOAD LOCAL LLM

llm = ChatOllama(model="llama3.1:8b",temperature=0)

# 2. DETECT CORRUPTED PDF CHUNKS

def is_corrupted_chunk(text):
    """
    Detect obviously garbled PDF extraction.

    Example:

        /g44/g47/g3/g19/...

    These chunks should not be sent to the LLM.
    """

    if not text:
        return True

    return text.count("/g") > 10

# 3. EXPAND RELATED EVIDENCE

def expand_related_chunks(reranked_results):
    """
    Expand only the strongest reranked result.

    This helps when retrieval finds the correct insurance
    form or endorsement, but the actual answer is stored
    in nearby chunks.

    Only nearby chunks from the same logical document
    or form number are retained.
    """

    expanded_documents = []
    seen = set()


    if not reranked_results:
        return expanded_documents


    # Use only the strongest reranked result.
    top_document = reranked_results[0]["document"]


    logical_document_id = top_document.metadata.get("logical_document_id")

    form_number = top_document.metadata.get("form_number")

    page = top_document.metadata.get("page")


    for chunk in final_chunks:

        chunk_logical_document_id = chunk.metadata.get("logical_document_id")

        chunk_form_number = chunk.metadata.get("form_number")

        chunk_page = chunk.metadata.get("page")


        # Same logical insurance document.
        same_logical_document = (
            logical_document_id is not None
            and
            chunk_logical_document_id == logical_document_id
        )


        # Or same insurance form.
        same_form = (
            form_number is not None
            and
            chunk_form_number == form_number
        )


        # Keep only nearby pages.
        nearby_page = (
            page is not None
            and
            chunk_page is not None
            and
            abs(chunk_page - page) <= 2
        )


        if (
            (same_logical_document or same_form)
            and nearby_page
        ):

            # Skip corrupted PDF extraction.
            if is_corrupted_chunk(
                chunk.page_content
            ):
                continue


            # Avoid duplicate chunks.
            chunk_key = (
                chunk_page,
                chunk.page_content
            )


            if chunk_key not in seen:

                expanded_documents.append(
                    chunk
                )

                seen.add(
                    chunk_key
                )


    return expanded_documents

# 4. RERANK EXPANDED EVIDENCE

def rerank_expanded_evidence(query,evidence_documents,final_k=6):
    """
    Rerank expanded same-form evidence against the
    original user query.

    This keeps the final LLM context focused on the
    strongest evidence chunks.
    """

    if not evidence_documents:
        return []

    # Build query-document pairs

    pairs = []


    for document in evidence_documents:

        pairs.append(
            [
                query,
                document.page_content
            ]
        )
    # Score expanded evidence

    scores = reranker.predict(pairs)


    scored_documents = []


    for index, document in enumerate(evidence_documents):

        scored_documents.append(
            {
                "document": document,
                "score": float(
                    scores[index]
                )
            }

    # Higher CrossEncoder score = better relevance.
    scored_documents = sorted(
        scored_documents,
        key=lambda item: item["score"],
        reverse=True
    )

    # Return only the strongest evidence documents.
    return [
        item["document"]
        for item in scored_documents[:final_k]
    ]

# 5. RETRIEVE + RERANK + EXPAND EVIDENCE

def get_evidence(query):
    """
    Full evidence retrieval pipeline:

        query
          ↓
        FAISS + BM25 hybrid retrieval
          ↓
        CrossEncoder reranking
          ↓
        focused same-form expansion
          ↓
        corrupted-text filtering
          ↓
        expanded evidence reranking
          ↓
        final evidence
    """

    # Hybrid candidate retrieval

    retrieved_results = retrieve_candidate(
        query,
        semantic_k=10,
        bm25_k=10
    )

    # Rerank hybrid candidates

    reranked_results = rerank_results(
        query,
        retrieved_results,
        final_k=4
    )

    # Expand strongest form/document

    expanded_documents = expand_related_chunks(
        reranked_results
    )

    # Rerank expanded evidence

    evidence_documents = rerank_expanded_evidence(
        query,
        expanded_documents,
        final_k=6
    )


    return evidence_documents

# 6. BUILD CONTEXT FOR THE LLM

def build_context(evidence_documents):
    """
    Convert retrieved LangChain Documents into a single
    grounded context string for the LLM.
    """

    context_parts = []


    for index, document in enumerate(
        evidence_documents,
        start=1
    ):

        form_number = document.metadata.get(
            "form_number"
        )

        logical_document_id = document.metadata.get(
            "logical_document_id"
        )

        document_type = document.metadata.get(
            "document_type"
        )

        parent_document_type = document.metadata.get(
            "parent_document_type"
        )

        section = document.metadata.get(
            "section"
        )

        page = document.metadata.get(
            "page"
        )

        text = document.page_content


        context_part = f"""
SOURCE {index}

Form number: {form_number}
Logical document ID: {logical_document_id}
Document type: {document_type}
Parent document type: {parent_document_type}
Section: {section}
Page: {page}

Policy text:
{text}
"""


        context_parts.append(
            context_part
        )


    return "\n".join(
        context_parts
    )

# 7. GENERATE GROUNDED ANSWER

def answer_question(query):
    """
    Generate an answer using only retrieved policy evidence.
    """

    evidence_documents = get_evidence(
        query
    )

    # If nothing useful was retrieved, do not call the LLM.

    if not evidence_documents:

        return (
            "I am unable to confirm this from the "
            "available policy evidence."
        )


    context = build_context(
        evidence_documents
    )


    prompt = f"""
You are an insurance policy document assistant.

Answer the user's question using ONLY the policy evidence
provided below.

Rules:

1. Use only the supplied policy evidence.

2. Do not use outside knowledge.

3. Do not guess or invent policy language.

4. Read the USER QUESTION carefully before answering.

5. Answer the exact question that was asked.

6. If the supplied evidence does not clearly answer the
   question, say exactly:

   "I am unable to confirm this from the available policy evidence."

7. If the evidence clearly answers the question, explain
   the answer briefly and clearly.

8. Mention the relevant form number, section, and page
   when available.

9. Do not make a final legal coverage determination.

10. Do not treat language from another insurance form as
    part of the requested endorsement unless the supplied
    evidence explicitly establishes that relationship.


USER QUESTION:

{query}


POLICY EVIDENCE:

{context}


ANSWER:
"""


    response = llm.invoke(
        prompt
    )


    return response.content

# 8. OPTIONAL RAG TEST

# Runs only when:
#
#     python rag.py
#
# Does not run when rag.py is imported by rag_evaluation.py.

if __name__ == "__main__":

    test_queries = [
        "What does the Auto Medical Payments endorsement cover?",
        "Who is an insured under the Auto Medical Payments endorsement?",
        "What is the limit of insurance for Auto Medical Payments Coverage?",
        "What does covered auto symbol 7 mean?",
        "Does this policy confirm that a rented truck is covered?"
    ]


    for query in test_queries:

        answer = answer_question(
            query
        )


        print(
            "\n" + "#" * 100
        )

        print(
            "QUESTION:"
        )

        print(
            query
        )

        print(
            "#" * 100
        )


        print(
            "\nRAG ANSWER:"
        )

        print(
            answer
        )