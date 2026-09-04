# Commercial Auto Insurance RAG
## Chunking Strategy Comparison and Reranking Impact Report

## 1. Purpose

This report evaluates the retrieval design used in the Commercial Auto Insurance RAG project.

The project requirement was to:

- implement two chunking strategies,
- compare retrieval quality using the same queries,
- add a reranking step,
- analyze whether reranking improves retrieval,
- and evaluate the final RAG pipeline.

The corpus consists of a 124-page commercial auto insurance forms and endorsements PDF.

The final system uses:

- Hugging Face embeddings
- FAISS semantic retrieval
- BM25 keyword retrieval
- Hybrid retrieval
- CrossEncoder reranking
- Same-form evidence expansion
- Corrupted-text filtering
- A second evidence reranking stage
- Ollama with Llama 3.1 8B for grounded answer generation


## 2. Evaluation Corpus

Source document:

`CAC-11Ed_FormsEndorsmnts.pdf`

Corpus statistics:

```text
Total PDF pages: 124
Processed documents: 124
Segmented documents: 124
```

Primary chunking strategy:

```text
Chunks before filtering: 469
Chunks after filtering: 395
```

The document contains multiple insurance-document types, including:

- Coverage forms
- Declarations
- Endorsements
- Coverage schedules
- Vehicle schedules
- Definitions and conditions


## 3. Chunking Strategy 1

### 3.1 Approach

Strategy 1 uses recursive text splitting.

Configuration:

```text
Chunk size: approximately 1000 characters
Chunk overlap: approximately 150 characters
```

The goal of this approach is to create reasonably sized chunks while preserving some context across chunk boundaries through overlap.

Implementation:

`src/chunking.py`


### 3.2 Advantages

Strategy 1 produced larger chunks that often retained more surrounding policy context.

This was useful for provisions where the meaning depended on several sentences appearing together.

It also produced a straightforward corpus for embedding and FAISS retrieval.


### 3.3 Limitations

Because chunk boundaries are primarily based on text length, they do not always align with insurance-document structure.

A heading may appear in one chunk while the substantive coverage language appears in another.

This issue later motivated evidence expansion.


## 4. Chunking Strategy 2

### 4.1 Approach

Strategy 2 was designed to be more structure-aware.

It attempts to preserve logical policy sections and headings instead of relying only on fixed character boundaries.

Implementation:

`src/chunking_strategy2.py`

The strategy created chunks with metadata such as:

```text
chunk_heading
form_number
document_type
parent_document_type
section
page
```

For example, the strategy separated headings such as:

```text
A. Description Of Covered Auto Designation Symbols
```

from surrounding document material while preserving section metadata.


### 4.2 Advantages

The second strategy created more logically focused chunks.

This helped isolate section headings and policy concepts and made the structure of retrieved evidence easier to inspect.

For Covered Auto Designation Symbols, the structure-aware strategy produced chunks specifically associated with the symbol-description section.


### 4.3 Limitations

Some chunks became too narrow.

For example, a retrieval result could contain only:

```text
A. Description Of Covered Auto Designation
Symbols
```

without the actual symbol definition needed to answer the question.

This demonstrated that smaller or more structurally precise chunks are not automatically better.

A chunk must preserve enough context to answer the user's question.


## 5. Chunking Comparison Method

Both chunking strategies were evaluated using the same set of questions.

Examples included:

1. What is the policy period?
2. What are hired autos?
3. What are non-owned autos?
4. Who is an insured under liability coverage?
5. What is a temporary substitute auto?
6. What does the Auto Medical Payments endorsement cover?
7. What does covered auto symbol 7 mean?

Implementation:

`src/retrieval_comparison.py`

The purpose was not simply to compare FAISS distance values.

The retrieved text was manually inspected for:

- relevance,
- completeness,
- preservation of policy context,
- useful metadata,
- and whether the chunk actually contained enough information to answer the question.


## 6. Chunking Comparison Results

The experiment showed that neither strategy was universally superior.

| Criterion | Strategy 1: Recursive | Strategy 2: Structure-Aware |
|---|---|---|
| Preserves surrounding context | Strong | Moderate |
| Preserves headings/sections | Moderate | Strong |
| Risk of cutting policy structure | Higher | Lower |
| Risk of overly small fragments | Lower | Higher |
| Ease of metadata inspection | Good | Strong |
| Useful for broad semantic retrieval | Strong | Good |
| Useful for exact section isolation | Good | Strong |

A notable example occurred with Covered Auto Designation Symbols.

Strategy 2 retrieved highly specific heading chunks such as:

`A. Description Of Covered Auto Designation Symbols`

but some of those chunks contained only the heading and not the actual symbol definition.

This showed that structural precision can sometimes reduce answer completeness.

The project therefore kept the primary recursive corpus for the final pipeline and added retrieval and evidence-expansion techniques to address its limitations.


## 7. Semantic Retrieval Evaluation

The initial retrieval method used FAISS semantic search.

FAISS is effective when the meaning of a query is similar to the policy text even when exact terms differ.

However, semantic retrieval showed limitations with identifiers and exact policy terminology.

### Example: Covered Auto Symbol 7

Query:

`What does covered auto symbol 7 mean?`

The top FAISS result returned material about:

```text
Symbol 70
Symbol 71
Symbol 79
```

rather than the exact Symbol 7 language.

This occurred because the embedding model correctly recognized that the chunk was semantically related to covered-auto designation symbols, but it did not strongly distinguish `7` from `70` or `71`.

This became a key motivation for adding BM25.


## 8. BM25 Keyword Retrieval

BM25 was added to improve lexical retrieval.

Implementation:

`src/retrieval.py`

BM25 is particularly useful for:

- numbers,
- exact identifiers,
- form names,
- coverage names,
- defined insurance terms.

For the Symbol 7 query, BM25 retrieved a Business Auto Coverage Form chunk that explicitly contained:

```text
if Symbol 7 is entered next to a coverage...
```

This result demonstrated that lexical retrieval could recover exact evidence that semantic similarity did not rank highly.

### Semantic vs. BM25 Observation

| Retrieval Method | Strength | Limitation |
|---|---|---|
| FAISS | Semantic intent and related concepts | Can confuse similar identifiers |
| BM25 | Exact wording, symbols, terminology | Can miss conceptual paraphrases |
| Hybrid | Combines both strengths | Requires additional ranking |

This experiment led to the final hybrid retrieval architecture.


## 9. Hybrid Retrieval

The final candidate-generation stage combines:

```text
FAISS top candidates
        +
BM25 top candidates
        ↓
deduplication
        ↓
combined candidate pool
```

Implementation:

`src/reranking.py`

The final configuration retrieves:

```text
FAISS semantic candidates: 10
BM25 keyword candidates: 10
```

Duplicate chunks are removed before reranking.

This design increases the chance that the correct evidence enters the candidate pool.


## 10. CrossEncoder Reranking

The project uses:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

The CrossEncoder evaluates the query and candidate chunk together.

Unlike FAISS distance or BM25 lexical scores, the reranker produces a new relevance score based on the relationship between the question and the retrieved text.

Implementation:

`src/reranking.py`

Evaluation:

`src/reranking_evaluation.py`


## 11. Reranking Impact

Reranking produced some of the clearest improvements in the project.

### 11.1 Temporary Substitute Auto

Query:

`What is a temporary substitute auto?`

Before reranking, the top semantic results included unrelated material such as pollution definitions and vehicle schedules.

After hybrid retrieval and CrossEncoder reranking, the top result became the actual temporary-substitute provision:

```text
Any "auto" you do not own while used with the permission
of its owner as a temporary substitute for a covered
"auto" you own that is out of service because of its:

a. Breakdown;
b. Repair;
c. Servicing;
d. "Loss"; or
e. Destruction.
```

This demonstrates a clear retrieval-quality improvement.

### 11.2 Covered Auto Symbol 7

Query:

`What does covered auto symbol 7 mean?`

Before reranking, the leading semantic candidates contained general Covered Auto Designation Symbol language.

After reranking, the BM25 result containing exact `Symbol 7` language moved to the top position.

This shows the benefit of:

```text
Hybrid retrieval
        +
CrossEncoder reranking
```

for exact insurance identifiers.


### 11.3 Auto Medical Payments

Query:

`What does the Auto Medical Payments endorsement cover?`

The semantic retriever already found:

```text
CA 99 03 10 13
AUTO MEDICAL PAYMENTS COVERAGE
```

The CrossEncoder preserved this as the strongest candidate.

BM25 also contributed supporting candidates such as declarations and additional Auto Medical Payments provisions.

This is an example where semantic retrieval was already strong and reranking maintained the correct result.


## 12. Why Reranking Alone Was Not Enough

The experiment also demonstrated an important RAG principle:

> A reranker can reorder candidates, but it cannot recover evidence that was never retrieved.

This is why the final project does not rely on:

```text
FAISS
→ reranker
```

alone.

Instead, it uses:

```text
FAISS
   +
BM25
   ↓
candidate pool
   ↓
CrossEncoder
```

Hybrid retrieval increases candidate recall, while reranking improves precision.


## 13. Evidence Expansion Experiment

Retrieval and reranking created another problem.

For the Auto Medical Payments query, the strongest candidate initially contained only:

```text
AUTO MEDICAL PAYMENTS COVERAGE
```

and introductory endorsement text.

The actual coverage provision appeared in a nearby chunk:

```text
A. Coverage

We will pay reasonable expenses incurred for
necessary medical and funeral services...
```

The system therefore introduced same-form evidence expansion.


### 13.1 Initial Expansion Failure

The first version expanded all four top reranked results.

This pulled dozens of chunks from multiple unrelated insurance forms into the LLM context.

The result was excessive context and poorer generation.


### 13.2 Focused Expansion

The final approach expands only the highest-ranked document using:

```text
logical_document_id
form_number
page
```

This successfully recovered related chunks from:

`CA 99 03 10 13`

without expanding unrelated forms.


## 14. Corrupted Text Filtering

The PDF contained some corrupted extracted chunks with patterns such as:

```text
/g44/g47/g3/g19/...
```

When these chunks were included in the prompt, the LLM became distracted and sometimes stated that the policy evidence was unreadable.

A corruption filter was therefore added before final context construction.

This improved generation quality for the Auto Medical Payments test.


## 15. Second-Stage Evidence Reranking

Same-form expansion can still produce several neighboring chunks.

Sending all of them to the LLM creates unnecessary context.

The project therefore applies the CrossEncoder a second time:

```text
expanded evidence
       ↓
CrossEncoder
       ↓
top 6 evidence chunks
```

This stage keeps the final prompt focused on the evidence most relevant to the original question.


## 16. Final Retrieval Architecture

The final retrieval architecture is:

```text
User Question
      |
      +------------------------+
      |                        |
      v                        v
FAISS Semantic             BM25 Keyword
Top 10                     Top 10
      |                        |
      +-----------+------------+
                  |
                  v
             Deduplicate
                  |
                  v
           CrossEncoder
              Rerank
                  |
                  v
           Best Candidate
                  |
                  v
        Same-Form Expansion
                  |
                  v
       Corrupted-Text Filter
                  |
                  v
         Evidence Reranking
                  |
                  v
        Top Evidence Chunks
                  |
                  v
              LLM
```


## 17. Final RAG Evaluation

The completed system was tested using 15 questions covering:

- Direct policy questions
- Definitions
- Endorsement questions
- Exact identifiers
- Multi-provision questions
- Ambiguous questions
- Questions requiring policy-specific information unavailable in the corpus

The final classification was:

| Result | Count |
|---|---:|
| Successful/useful grounded responses | 10 |
| Safe or conditional refusals | 3 |
| Partial/retrieval-limited results | 2 |
| Total questions | 15 |

The final test therefore demonstrated both successful retrieval and failure handling.


## 18. Final Evaluation Examples

### Successful — Auto Medical Payments

The system correctly retrieved:

```text
CA 99 03 10 13
A. Coverage
```

and answered that Auto Medical Payments covers reasonable medical and funeral expenses for an insured who sustains bodily injury caused by an accident.

### Successful — Temporary Substitute Auto

The system correctly identified:

- non-owned auto,
- owner's permission,
- covered owned auto out of service,
- breakdown,
- repair,
- servicing,
- loss,
- or destruction.

### Successful — Symbol 1

The system correctly retrieved:

```text
1 Any "Auto"
```

### Ambiguous — Rented Trucks

The retrieved evidence stated that rented or hired autos may be treated as covered autos under the relevant provision, while autos rented with a driver may be treated differently.

Because the query did not specify driver status, the system appropriately avoided an unconditional answer.

### Safe Refusal — VIN-Specific Coverage

The question:

```text
Was vehicle VIN 1ABC23456789 covered on January 15, 2024?
```

could not be answered because the forms-library corpus did not contain issued-policy evidence tying that VIN to coverage on the requested date.

The system correctly returned an unable-to-confirm response.


## 19. Failure Analysis

### 19.1 Symbol 7

BM25 found relevant Symbol 7 language and the CrossEncoder promoted it during reranking.

However, the final evidence-selection process still did not produce sufficiently complete evidence for the final RAG answer.

The system therefore refused to answer.

This illustrates that:

```text
good retrieval
does not automatically mean
good final evidence selection
```

### 19.2 Auto Medical Payments Limit

The system located a declarations schedule showing:

```text
Auto Medical Payments
$$
Each Insured
```

but the source contained no readable dollar value.

The pipeline therefore found the correct field but could not provide the actual amount.

### 19.3 Medical Payments for an Occupant

The system retrieved an Auto Medical Payments schedule and generated a plausible answer.

However, stronger evidence existed in the endorsement's:

`B. Who Is An Insured`

section.

This is an example of a relevant retrieval result that was not the strongest possible supporting source.


## 20. Main Findings

### Finding 1 — Chunk structure matters

Structure-aware chunking improved section preservation but sometimes created fragments that were too small to answer a question.

Recursive chunking retained more surrounding context but sometimes separated headings from substantive provisions.

### Finding 2 — Semantic search and keyword search solve different problems

FAISS performs well for semantic meaning.

BM25 performs better for exact terms and identifiers.

### Finding 3 — Hybrid retrieval improves recall

Combining FAISS and BM25 increases the chance that relevant evidence enters the candidate pool.

### Finding 4 — Reranking improves precision

The CrossEncoder successfully promoted exact and highly relevant policy evidence above weaker initial candidates.

### Finding 5 — Parent/form-aware expansion matters

Insurance provisions often span multiple chunks.

Same-form evidence expansion allowed the system to recover complete coverage provisions.

### Finding 6 — Too much context hurts

Expanding multiple forms produced noisy prompts.

Focused expansion and second-stage reranking improved context quality.

### Finding 7 — Source quality affects RAG quality

Malformed PDF extraction can directly reduce LLM answer quality.

### Finding 8 — Refusal behavior is necessary

A grounded insurance RAG system should identify insufficient evidence instead of inventing policy-specific facts.


## 21. Final Conclusion

The experiments demonstrate that retrieval quality cannot be optimized using one technique alone.

The strongest final pipeline combined:

```text
Recursive chunking
        +
insurance metadata
        +
MiniLM embeddings
        +
FAISS semantic retrieval
        +
BM25 exact-term retrieval
        +
CrossEncoder reranking
        +
same-form evidence expansion
        +
corrupted-text filtering
        +
second-stage evidence reranking
```

The chunking comparison showed the tradeoff between preserving document structure and retaining enough context.

The retrieval experiments showed that semantic search alone can miss exact insurance identifiers, while BM25 can retrieve those identifiers effectively.

The reranking experiments showed clear improvements in ordering relevant evidence, particularly for temporary substitute autos and Covered Auto Symbol 7.

Finally, the 15-question RAG evaluation demonstrated both successful grounded answers and appropriate failure handling.

The final design therefore prioritizes not only retrieval relevance but also evidence completeness, source quality, and conservative generation.