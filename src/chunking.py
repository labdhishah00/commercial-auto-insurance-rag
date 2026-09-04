#---Phase 7: Chunking---#

from ingestion import segmented_documents #file from ingestion#
from langchain_text_splitters import RecursiveCharacterTextSplitter #to divide large pieces of text into smaller chunks#

#---create text splitter---#
# \n\n - paragraph boundaries, \n - line, . - sentences like, " " -individual words, "" - final characters if no better split exists

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

#---create raw chunks---#

chunks = text_splitter.split_documents(segmented_documents)

#---filter useless page-number chunks---#

final_chunks = []

for chunk in chunks:
    chunk_text = chunk.page_content.strip()

    page_label = str(chunk.metadata.get("page_label", "")).strip()

    is_page_number_chunk = (chunk_text.isdigit() and len(chunk_text) <= 4 and chunk_text == page_label)

    if not is_page_number_chunk:
        final_chunks.append(chunk)

print("Total segmented documents:", len(segmented_documents))
print("Chunks before filtering:", len(chunks))
print("Chunks after filtering:", len(final_chunks))