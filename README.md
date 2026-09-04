# Commercial Auto Insurance RAG System

## 1. Project Overview

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system for commercial auto insurance policy documents.

The system retrieves relevant policy language from commercial auto coverage forms, declarations, schedules, and endorsements and generates answers grounded only in the retrieved policy evidence.

The project explores and evaluates multiple RAG techniques, including:

- PDF document ingestion
- Insurance-specific metadata extraction
- Logical document segmentation
- Multiple chunking strategies
- Hugging Face sentence embeddings
- FAISS semantic retrieval
- BM25 keyword retrieval
- Hybrid retrieval
- CrossEncoder reranking
- Same-form evidence expansion
- Corrupted-text filtering
- Expanded-evidence reranking
- Grounded answer generation using a local LLM through Ollama
- Safe refusal when sufficient evidence is unavailable

The final system is designed to assist with policy research and coverage review. It does not make a final legal coverage determination.


## 2. Business Problem

Commercial auto insurance documents may contain:

- Declarations
- Coverage forms
- Endorsements
- Vehicle schedules
- Definitions
- Conditions
- Exclusions
- Coverage symbols
- Limits of insurance

Finding the correct policy provision manually can be time-consuming.

For a claim adjuster or insurance analyst, answering a seemingly simple question may require locating the correct coverage form, identifying an endorsement, reviewing definitions, and determining whether the available policy evidence is sufficient.

The goal of this project is to build a RAG-based assistant that can:

1. Search commercial auto insurance documents.
2. Retrieve policy language relevant to a user's question.
3. Preserve important metadata such as form number, section, document type, and page.
4. Combine semantic and keyword retrieval.
5. Rerank retrieved evidence according to relevance.
6. Recover related evidence when a policy provision spans multiple chunks.
7. Generate answers using only the retrieved evidence.
8. Avoid unsupported conclusions when the evidence is incomplete.

When sufficient evidence is unavailable, the system is designed to respond conservatively rather than guess.


## 3. Dataset

The project uses the following commercial auto insurance forms and endorsements document:

`CAC-11Ed_FormsEndorsmnts.pdf`

The PDF contains:

```text
Total PDF pages: 124
Total processed documents: 124
Total segmented documents: 124
```

For the primary chunking strategy:

```text
Chunks before filtering: 469
Chunks after filtering: 395
```

The project preserves insurance-specific metadata where available, including:

- `form_number`
- `document_type`
- `parent_document_type`
- `logical_document_id`
- `section`
- `page`

This metadata is later used for:

- Retrieval inspection
- Source identification
- Same-form evidence expansion
- Context construction
- Grounded answer generation


## 4. Project Architecture

The final RAG pipeline follows this architecture:

```text
                 Commercial Auto Insurance PDF
                              |
                              v
                       PDF Ingestion
                              |
                              v
                  Document Processing
                       + Metadata
                              |
                              v
                     Segmentation
                              |
                              v
                        Chunking
                              |
                              v
                       Embeddings
                  all-MiniLM-L6-v2
                              |
                              v
                    Hybrid Retrieval
                       /           \
                      v             v
                  FAISS            BM25
               Semantic Search  Keyword Search
                      \             /
                       \           /
                        v         v
                     Candidate Merge
                     + Deduplication
                              |
                              v
                       CrossEncoder
                         Reranking
                              |
                              v
                     Strongest Candidate
                              |
                              v
                Same-Form / Logical-Document
                     Evidence Expansion
                              |
                              v
                  Corrupted Chunk Filtering
                              |
                              v
                 Expanded Evidence Reranking
                              |
                              v
                    Top Evidence Chunks
                              |
                              v
                         RAG Prompt
                              |
                              v
                   Ollama / Llama 3.1 8B
                              |
                              v
                      Grounded Answer
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
           Evidence sufficient     Evidence insufficient
                  |                       |
                  v                       v
           Grounded explanation    Unable-to-confirm
```

The final architecture combines multiple retrieval methods because different insurance questions require different retrieval behavior.

FAISS is useful for semantic similarity, while BM25 is useful for exact terminology, identifiers, coverage symbols, and policy phrases.

CrossEncoder reranking then improves the ordering of the combined candidate set.


## 5. Document Processing and Chunking Strategies

### 5.1 Document Ingestion

The source PDF is loaded using `PyPDFLoader`.

The ingestion pipeline:

1. Locates the PDF using the project root directory.
2. Loads all 124 pages.
3. Extracts page text.
4. Preserves original page metadata.
5. Adds insurance-specific metadata for downstream processing.

The ingestion logic is implemented in:

`src/ingestion.py`


### 5.2 Logical Document Segmentation

After ingestion, the pages are processed into logical insurance documents.

The segmentation stage identifies information such as:

- Form number
- Document type
- Parent document type
- Logical document ID
- Section
- Page

Example metadata:

```text
form_number: CA 99 03 10 13
document_type: endorsement
parent_document_type: endorsement
logical_document_id: logical_doc_3
page: 34
```

This structure becomes especially important later when the system expands evidence from the same insurance form.


### 5.3 Chunking Strategy 1

The primary chunking strategy uses recursive text splitting.

Configuration:

```text
Chunk size: 1000 characters
Chunk overlap: 150 characters
```

Chunk overlap helps reduce the chance that important policy provisions are lost when a section crosses a chunk boundary.

Results:

```text
Chunks before filtering: 469
Chunks after filtering: 395
```

Implementation:

`src/chunking.py`


### 5.4 Chunking Strategy 2

A second chunking approach was created to evaluate a more structure-aware strategy.

Rather than relying only on fixed character boundaries, this strategy attempts to preserve more of the logical structure of policy sections and forms.

Implementation:

`src/chunking_strategy2.py`

A separate vector store was also created for this strategy:

`src/vector_store_strategy2.py`


### 5.5 Chunking Strategy Comparison

Both strategies were tested using the same insurance questions.

Example comparison questions included:

- What is the policy period?
- What are hired autos?
- What are non-owned autos?
- Who is an insured under liability coverage?
- What is a temporary substitute auto?
- What does the Auto Medical Payments endorsement cover?
- What does covered auto symbol 7 mean?

The comparison is implemented in:

`src/retrieval_comparison.py`

The experiment demonstrated that chunking strategy affects retrieval quality, but changing chunking alone does not solve every retrieval problem.

This motivated the addition of hybrid semantic and lexical retrieval.


## 6. Embeddings and Vector Store

### 6.1 Embedding Model

The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

The model converts policy chunks into numerical embedding vectors representing semantic meaning.

The embedding dimension is:

```text
384
```

For the primary corpus:

```text
Final chunks: 395
Embedding vectors: 395
Embedding dimensions: 384
```

Embedding configuration is implemented in:

`src/embeddings.py`


### 6.2 FAISS Vector Store

FAISS is used as the semantic vector store.

Each final policy chunk is stored together with its metadata.

During semantic retrieval, the user's question is converted into an embedding and compared against the stored policy vectors.

Implementation:

`src/vector_store.py`

Strategy 2 uses:

`src/vector_store_strategy2.py`


## 7. Hybrid Retrieval

The project initially used FAISS semantic search.

Semantic search performed well for conceptually similar language, but testing showed limitations with exact insurance terminology, identifiers, numerical symbols, and policy-specific phrases.

The final system therefore combines:

- FAISS semantic retrieval
- BM25 keyword retrieval


### 7.1 FAISS Semantic Search

FAISS helps retrieve conceptually related text even when the wording of the question is different from the policy.

For example, a question involving rented vehicles may retrieve text containing terms such as:

```text
lease
hire
rent
borrow
```

Semantic retrieval is implemented through:

`src/vector_store.py`

and:

`src/retrieval.py`


### 7.2 BM25 Keyword Search

BM25 operates on tokens instead of embeddings.

It is useful for exact insurance terminology such as:

- Coverage names
- Defined terms
- Form numbers
- Covered auto symbols
- Specific phrases

During testing, the query:

`What does covered auto symbol 7 mean?`

showed an important difference between the two retrieval approaches.

FAISS returned semantically related Covered Auto Designation Symbol sections, but its highest-ranked results were not necessarily the exact Symbol 7 provision.

BM25 retrieved policy language explicitly containing `Symbol 7`.

This demonstrated why semantic and lexical retrieval complement one another.


### 7.3 Hybrid Candidate Generation

The final retrieval process is:

```text
                   User Question
                         |
              +----------+----------+
              |                     |
              v                     v
        FAISS Semantic          BM25 Keyword
            Search                 Search
              |                     |
              +----------+----------+
                         |
                         v
                 Candidate Merge
                         |
                         v
                   Deduplication
                         |
                         v
                 CrossEncoder
                    Reranking
```

The candidate set from both methods is merged.

Duplicate chunks are removed before reranking.

Implementation:

`src/retrieval.py`

Hybrid candidate generation and reranking are integrated in:

`src/reranking.py`


## 8. CrossEncoder Reranking

The final project uses:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The CrossEncoder evaluates the question and a candidate chunk together.

Conceptually:

```text
Question + Candidate Chunk
           |
           v
      CrossEncoder
           |
           v
     Relevance Score
```

Higher scores indicate stronger predicted relevance.


### 8.1 Initial Reranking

FAISS and BM25 produce candidate chunks.

The CrossEncoder then reranks the combined candidate pool.

Implementation:

`src/reranking.py`


### 8.2 Before-vs-After Evaluation

Reranking was tested using the same question set used during earlier retrieval experiments.

The evaluation compares:

```text
Before reranking
Hybrid FAISS + BM25 candidates

vs.

After reranking
CrossEncoder top candidates
```

Implementation:

`src/reranking_evaluation.py`

The evaluation showed that reranking can significantly improve candidate ordering.

For example, for a temporary substitute auto question, weaker semantic candidates appeared before reranking, while BM25 candidates containing the actual temporary-substitute provision moved to the top after CrossEncoder reranking.

For the Symbol 7 test, the exact BM25 result was promoted to the top candidate after reranking.


### 8.3 Limitation of Reranking

Reranking can only reorder candidates that were already retrieved.

It cannot recover evidence that never entered the candidate pool.

This is one reason hybrid candidate generation was used instead of FAISS alone.


## 9. Evidence Expansion and Quality Filtering

### 9.1 Evidence Expansion

Insurance provisions frequently span multiple chunks.

For example, retrieval may identify:

`AUTO MEDICAL PAYMENTS COVERAGE`

while the actual:

`A. Coverage`

language appears in another nearby chunk.

To address this problem, the system expands the strongest reranked candidate using:

- `logical_document_id`
- `form_number`
- `page`

This allows nearby chunks from the same insurance form or logical document to be included.


### 9.2 Focused Expansion

An early experiment expanded all top reranked documents.

This created excessive and unrelated context.

The final implementation therefore expands only the strongest reranked form/document.


### 9.3 Corrupted PDF Filtering

Some PDF extraction produced corrupted strings such as:

```text
/g44/g47/g3/g19/...
```

These chunks reduced generation quality.

The final RAG pipeline identifies and excludes obviously corrupted chunks before generation.


### 9.4 Expanded Evidence Reranking

Even after same-form expansion, not every nearby chunk is equally relevant.

The expanded chunks are therefore reranked again using the CrossEncoder.

Only the strongest evidence chunks are sent to the LLM.

The final evidence workflow is:

```text
Hybrid Retrieval
       |
       v
Initial Reranking
       |
       v
Strongest Form
       |
       v
Same-Form Expansion
       |
       v
Corruption Filtering
       |
       v
Evidence Reranking
       |
       v
Top Evidence
```


## 10. RAG Answer Generation

The final evidence is passed to a local Large Language Model.

Model:

`llama3.1:8b`

Runtime:

`Ollama`

Implementation:

`src/rag.py`


### 10.1 Grounded Prompting

The model is instructed to:

- Use only the supplied policy evidence.
- Avoid outside knowledge.
- Avoid guessing.
- Answer the exact question asked.
- Mention form number, section, and page when available.
- Avoid making a final legal coverage determination.

If the evidence does not clearly answer the question, the model is instructed to say:

> "I am unable to confirm this from the available policy evidence."


### 10.2 Final Answer-Generation Pipeline

```text
User Question
      |
      v
FAISS + BM25
      |
      v
Deduplication
      |
      v
CrossEncoder Reranking
      |
      v
Strongest Candidate
      |
      v
Same-Form Expansion
      |
      v
Corrupted-Text Filtering
      |
      v
Expanded Evidence Reranking
      |
      v
Top Evidence Chunks
      |
      v
Grounded Prompt
      |
      v
Ollama / Llama 3.1 8B
      |
      v
Grounded Answer
```


## 11. Final Evaluation

The final pipeline was stress-tested using 15 questions.

Implementation:

`src/rag_evaluation.py`

The evaluation covers:

- Direct policy questions
- Definitions
- Endorsement questions
- Exact identifiers
- Multi-provision questions
- Ambiguous coverage questions
- Questions where policy-specific evidence is unavailable


### 11.1 Evaluation Questions

The 15 evaluation questions were:

1. What is the policy period?
2. What are hired autos?
3. What are non-owned autos?
4. Who is an insured under liability coverage?
5. What is a temporary substitute auto?
6. What does the Auto Medical Payments endorsement cover?
7. Who is an insured under the Auto Medical Payments endorsement?
8. What is the limit of insurance for Auto Medical Payments Coverage?
9. What does covered auto symbol 7 mean?
10. What does covered auto symbol 1 mean?
11. When can a non-owned auto be used as a temporary substitute for a covered auto?
12. Are medical payments available for someone occupying a covered auto?
13. Are rented trucks covered?
14. Is every vehicle owned by the insured automatically covered?
15. Was vehicle VIN 1ABC23456789 covered on January 15, 2024?


### 11.2 Evaluation Summary

| Result Category | Count |
|---|---:|
| Successful or useful grounded responses | 10 |
| Safe or conditional refusals | 3 |
| Partial / retrieval-limited responses | 2 |
| Total | 15 |

The evaluation demonstrates that the system performs well for direct definitions, policy provisions, and endorsement questions while remaining conservative when policy-specific evidence is unavailable.


### 11.3 Successful Examples

#### Auto Medical Payments Coverage

Question:

`What does the Auto Medical Payments endorsement cover?`

The system retrieved form:

`CA 99 03 10 13`

and the relevant `A. Coverage` provision.

The answer correctly explained that the endorsement covers reasonable expenses for necessary medical and funeral services for an insured who sustains bodily injury caused by an accident.


#### Auto Medical Payments — Who Is An Insured

The system successfully retrieved the endorsement provision identifying:

- The named insured while occupying an auto or when struck as a pedestrian
- Family members under the applicable conditions
- Other persons occupying a covered auto or temporary substitute auto


#### Temporary Substitute Auto

The system retrieved policy language stating that a non-owned auto may serve as a temporary substitute when used with the owner's permission while the covered owned auto is out of service because of:

- Breakdown
- Repair
- Servicing
- Loss
- Destruction


#### Covered Auto Symbol 1

The system correctly retrieved:

`Any "Auto"`


### 11.4 Ambiguous Coverage Example

Question:

`Are rented trucks covered?`

The system retrieved language explaining that an auto that is leased, hired, rented, or borrowed may be treated as a covered auto under the relevant provision, but an auto rented with a driver may be treated differently.

Because the question did not state whether the rented vehicle included a driver, the system did not make an unconditional coverage determination.

This demonstrates appropriate handling of ambiguity.


### 11.5 Safe Refusal Example

Question:

`Was vehicle VIN 1ABC23456789 covered on January 15, 2024?`

The source corpus does not contain issued-policy information tying that VIN to coverage on that date.

The system correctly responded that it could not confirm coverage from the available evidence.


### 11.6 Partial Result: Auto Medical Payments Limit

Question:

`What is the limit of insurance for Auto Medical Payments Coverage?`

The system retrieved a declarations schedule containing:

```text
Auto Medical Payments
$$
Each Insured
```

However, the extracted evidence did not contain a usable dollar amount.

This result is classified as partial because the relevant field was located, but the value could not be reliably determined.


### 11.7 Partial Result: Medical Payments for an Occupant

Question:

`Are medical payments available for someone occupying a covered auto?`

The system retrieved a coverage schedule mentioning Auto Medical Payments.

The answer was plausible, but the strongest supporting evidence should ideally have been the Auto Medical Payments endorsement's `Who Is An Insured` provision.

This demonstrates that retrieving a related coverage schedule is not always equivalent to retrieving the strongest evidence for the exact question.


### 11.8 Symbol 7 Limitation

Question:

`What does covered auto symbol 7 mean?`

BM25 successfully retrieved language explicitly referring to Symbol 7 during the retrieval and reranking experiments.

However, the final evidence-selection path did not preserve sufficiently clear evidence to confidently answer the complete definition question.

The final RAG therefore returned an unable-to-confirm response instead of inventing an answer.

This is an important finding because it demonstrates that successful intermediate retrieval does not guarantee that the correct evidence survives every downstream stage.


## 12. Key Evaluation Findings

The experiments produced several important lessons.

### 12.1 Semantic Retrieval Alone Is Not Enough

FAISS performs well for conceptual similarity but may return related policy language instead of the exact identifier requested.

### 12.2 BM25 Improves Exact-Term Retrieval

BM25 is valuable for terms such as:

```text
Symbol 7
Auto Medical Payments
temporary substitute auto
form identifiers
```

### 12.3 Hybrid Retrieval Improves Candidate Coverage

Combining semantic and keyword retrieval creates a stronger candidate pool than relying on either technique alone.

### 12.4 Reranking Improves Candidate Ordering

The CrossEncoder frequently promoted more relevant BM25 or semantic candidates above weaker initial matches.

### 12.5 Evidence Expansion Solves Fragmentation

The Auto Medical Payments test demonstrated that finding only the endorsement title was insufficient.

Expanding the same form recovered the substantive coverage provision.

### 12.6 Excessive Expansion Hurts Generation

Expanding multiple unrelated forms created noisy LLM context.

The final pipeline therefore expands only the strongest form/document.

### 12.7 PDF Quality Matters

Corrupted extracted text distracted the LLM.

Filtering low-quality chunks improved the final context.

### 12.8 Safe Refusal Is a Feature

A RAG system used for insurance research should not invent coverage facts when policy evidence is missing.

The VIN-specific test demonstrates this behavior.


## 13. Limitations

### 13.1 Retrieval Is Not Perfect

Even hybrid retrieval and reranking do not guarantee that the ideal evidence reaches the final LLM context.

The Symbol 7 test demonstrates this limitation.


### 13.2 Source Document Is a Forms Library

The source PDF contains commercial auto forms, declarations examples, schedules, and endorsements rather than a complete issued policy for one specific insured.

The system therefore cannot reliably determine policy-specific facts such as:

- Actual effective and expiration dates
- Selected coverage symbols
- Actual limits
- Scheduled vehicles
- Specific drivers
- Applied endorsements
- Whether a particular vehicle was insured on a particular loss date


### 13.3 PDF Extraction Quality

Some pages contain malformed extracted text.

Basic corruption filtering was added, but a production implementation would benefit from a stronger document parser.


### 13.4 Coverage Determination

The system assists with policy research.

It does not provide a final legal or claim coverage determination.


### 13.5 Local Runtime Cost

The project loads:

- Embedding models
- FAISS
- CrossEncoder
- Local Llama 3.1 8B

Execution time may therefore depend on the user's hardware.


## 14. Project Structure

```text
insurance-rag/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── CAC-11Ed_FormsEndorsmnts.pdf
│
└── src/
    ├── ingestion.py
    ├── chunking.py
    ├── chunking_strategy2.py
    ├── embeddings.py
    ├── vector_store.py
    ├── vector_store_strategy2.py
    ├── retrieval.py
    ├── retrieval_comparison.py
    ├── reranking.py
    ├── reranking_evaluation.py
    ├── rag.py
    └── rag_evaluation.py
```


## 15. File Responsibilities

### `ingestion.py`

Loads the PDF, extracts text, and creates insurance-specific metadata.

### `chunking.py`

Implements the primary recursive chunking strategy.

### `chunking_strategy2.py`

Implements the second chunking strategy used for comparison.

### `embeddings.py`

Configures the Hugging Face embedding model.

### `vector_store.py`

Builds the FAISS vector store for the primary chunk corpus.

### `vector_store_strategy2.py`

Builds the vector store used for the second chunking strategy.

### `retrieval.py`

Implements FAISS semantic retrieval and BM25 keyword retrieval.

### `retrieval_comparison.py`

Compares retrieval behavior across chunking strategies.

### `reranking.py`

Combines FAISS and BM25 candidates, removes duplicates, and applies CrossEncoder reranking.

### `reranking_evaluation.py`

Compares hybrid candidates before and after CrossEncoder reranking.

### `rag.py`

Implements the final answer-generation pipeline including:

- Hybrid retrieval
- Reranking
- Evidence expansion
- Corruption filtering
- Expanded-evidence reranking
- Context construction
- Ollama generation
- Safe refusal behavior

### `rag_evaluation.py`

Runs the final 15-question stress test.


## 16. Installation and Setup

### 16.1 Python

The project was developed using Python 3.12.


### 16.2 Install Dependencies

From the project root:

```powershell
python -m pip install -r requirements.txt
```

The project requirements include:

```text
langchain
langchain-community
langchain-huggingface
langchain-ollama
pypdf
sentence-transformers
faiss-cpu
rank-bm25
scikit-learn
```


### 16.3 Install Ollama

Ollama must be installed separately because it is not a Python package.

Verify the installation:

```powershell
ollama --version
```

Download the local model:

```powershell
ollama pull llama3.1:8b
```

Optional direct model test:

```powershell
ollama run llama3.1:8b
```


### 16.4 Source PDF

Place the PDF in:

```text
insurance-rag/data/
```

Expected file:

```text
CAC-11Ed_FormsEndorsmnts.pdf
```


## 17. How to Run

From the project root:

```powershell
cd src
```


### Run Primary Chunking

```powershell
python chunking.py
```


### Run Strategy 2 Chunking

```powershell
python chunking_strategy2.py
```


### Run Primary Vector Store

```powershell
python vector_store.py
```


### Run Retrieval Demonstration

```powershell
python retrieval.py
```


### Run Chunking Strategy Comparison

```powershell
python retrieval_comparison.py
```


### Run Reranking Evaluation

```powershell
python reranking_evaluation.py
```


### Run the RAG Pipeline

```powershell
python rag.py
```


### Run the Final 15-Question Evaluation

```powershell
python rag_evaluation.py
```


## 18. Future Improvements

### 18.1 Policy-Specific Corpus

Add complete issued-policy packages containing:

- Declarations
- Applied coverage forms
- Applied endorsements
- Vehicle schedules
- Driver schedules
- Actual policy limits
- Effective and expiration dates

This would allow the system to answer more policy-specific questions.


### 18.2 Metadata-Aware Retrieval

Future retrieval could apply filters or boosts based on:

- Form number
- Document type
- Coverage form
- Endorsement
- Coverage symbol
- Policy period
- Vehicle
- Driver


### 18.3 Reciprocal Rank Fusion

Rather than simply merging semantic and BM25 candidates, future versions could use Reciprocal Rank Fusion to combine retrieval rankings more systematically.


### 18.4 Improved PDF Parsing

A stronger parser could better preserve:

- Tables
- Multi-column forms
- Headings
- Schedules
- Form boundaries
- Scanned pages


### 18.5 Query Classification

The system could classify questions before retrieval.

Examples:

```text
definition question
coverage question
identifier question
vehicle-specific question
policy-date question
endorsement question
```

Different retrieval strategies could then be selected depending on question type.


### 18.6 Coverage Evidence Classification

Future versions could classify retrieved evidence into outcomes such as:

```text
CLEAR SUPPORT
CLEAR EXCLUSION
INSUFFICIENT EVIDENCE
CONFLICTING EVIDENCE
```

This could better support claim-adjuster workflows while still avoiding unsupported coverage conclusions.


### 18.7 Claim-Specific Inputs

Future versions could incorporate:

- VIN
- Driver information
- Date of loss
- Policy period
- Coverage symbols
- Vehicle ownership
- Rental status
- Applied endorsements

Exact identifiers could be combined with semantic policy retrieval.


## 19. Conclusion

This project demonstrates an end-to-end Retrieval-Augmented Generation system for commercial auto insurance documents.

The final pipeline combines:

```text
Document ingestion
        +
insurance metadata
        +
logical segmentation
        +
chunking experiments
        +
MiniLM embeddings
        +
FAISS semantic retrieval
        +
BM25 lexical retrieval
        +
hybrid candidate generation
        +
CrossEncoder reranking
        +
same-form evidence expansion
        +
corrupted-text filtering
        +
expanded evidence reranking
        +
grounded Llama 3.1 generation
```

The experiments showed that no single retrieval method solved every insurance question.

Semantic search was useful for conceptual similarity.

BM25 improved exact-term and identifier retrieval.

CrossEncoder reranking improved candidate ordering.

Evidence expansion helped recover provisions divided across multiple chunks.

Filtering corrupted PDF text improved LLM context quality.

The final 15-question evaluation also demonstrated an important property of the system: when the available policy evidence is insufficient, the assistant can decline to make an unsupported coverage conclusion.

The project therefore demonstrates not only a working RAG pipeline, but also the retrieval, evaluation, grounding, and failure-analysis techniques needed to build a more reliable document intelligence system.