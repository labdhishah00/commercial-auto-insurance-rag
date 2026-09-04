#Phase : Chunking part 2 - structure-aware/section-aware chunking

from ingestion import segmented_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re

#define heading patterns

heading_pattern = re.compile(r'(?m)^('r'SECTION\S+[IVX]+\b[^\n]*'r'|'r'ITEM\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN)\s*'r'|'r'[A-Z]\.\s+[A-Z][^\n]*'r')$', flags=re.IGNORECASE)

#Fallback splitter

fallback_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 150, separators = ["\n\n","\n",". "," ",""])

def merge_small_heading_chunks(section_documents):
    merged_documents = []
    index = 0

    while index < len(section_documents):
        current_doc = section_documents[index]
        current_text = current_doc.page_content.strip()
        current_heading = current_doc.metadata.get("chunk_heading")

        is_small_heading_chunk = (len(current_text) < 150 and current_heading is not None and current_heading != "DOCUMENT INTRODUCTION" and index + 1 < len(section_documents))

        if is_small_heading_chunk:
            next_doc = section_documents[index + 1]
            combined_text = (current_text + "\n" + next_doc.page_content.strip())
            combined_metadata = (next_doc.metadata.copy())
            combined_metadata["chunk_heading"] = (current_heading)
            combined_doc = Document(page_content = combined_text, metadata = combined_metadata)
            merged_documents.append(combined_doc)
            index += 2
        else:
            merged_documents.append(current_doc)
            index += 1
        return merged_documents



#split one document by headings

def split_by_structure(doc):
    text = doc.page_content
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return fallback_splitter.split_documents([doc])
    
    #create section ranges

    section_documents = []

    for index, match in enumerate(matches):
        start = match.start()

        if index +1 < len(matches):
            end = matches[index +1].start()
        
        else:
            end = len(text)
        section_text = text[start:end].strip()

        if not section_text:
            continue
        
        metadata = doc.metadata.copy()
        heading = match.group(0).strip()

        metadata["chunk_heading"] = heading

        section_doc = Document(page_content = section_text, metadata=metadata)
        section_documents.append(section_doc)
    
    first_heading_start = matches[0].start()

    if first_heading_start > 0:
        prefix_text = text[:first_heading_start].strip()


        if prefix_text:

            prefix_metadata = (doc.metadata.copy())

            prefix_metadata["chunk_heading"] = "DOCUMENT INTRODUCTION"


            prefix_doc = Document(page_content=prefix_text,metadata=prefix_metadata)
            section_documents.insert(0, prefix_doc)
    
    section_documents = merge_small_heading_chunks(section_documents)

    final_section_chunks = []


    for section_doc in section_documents:


        # If the structure-aware section is already small,
        # preserve it exactly.
        if len(section_doc.page_content) <= 1200:

            final_section_chunks.append(section_doc)
        else:

            # If section is too long, recursively split ONLY that section.
            smaller_chunks = (fallback_splitter.split_documents([section_doc]))
            final_section_chunks.extend(smaller_chunks)


    return final_section_chunks


strategy2_chunks = []


for doc in segmented_documents:

    page_chunks = split_by_structure(doc)

    strategy2_chunks.extend(page_chunks)


final_strategy2_chunks = []


for chunk in strategy2_chunks:

    chunk_text = (chunk.page_content.strip())

    page_label = str(chunk.metadata.get("page_label","")).strip()


    is_page_number_chunk = (chunk_text.isdigit() and len(chunk_text) <= 4 and chunk_text == page_label)


    if not is_page_number_chunk:
        final_strategy2_chunks.append(chunk)


print("Total segmented documents:", len(segmented_documents))

print("Strategy 2 chunks before filtering:", len(strategy2_chunks))

print("Strategy 2 chunks after filtering:", len(final_strategy2_chunks))
