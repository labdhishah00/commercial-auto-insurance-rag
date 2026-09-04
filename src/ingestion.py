from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import re

#---Phase 1: Load the pdf---#

print("SCRIPT STARTED")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pdf_path = PROJECT_ROOT / "data" / "CAC-11Ed_FormsEndorsmnts.pdf"

print("Project root:", PROJECT_ROOT)
print("PDF path:", pdf_path)
print("PDF exists:", pdf_path.exists())

loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

print("Total pages:", len(documents))

#---Phase 2: Cleaning Function---#

def clean_text(text):
    
    text = re.sub(r'Commercial[ ]+Auto[ ]+Coverage,[ ]+11th[ ]+Edition','',text,flags=re.IGNORECASE)   #remove repeated edition footer
    text = re.sub(r'CA[ ]+[A-Z0-9 ]+[ ]+©[ ]+Insurance[ ]+Services'r'[ ]+Office,[ ]+Inc\.,[ ]+\d{4}[ ]+Page'+r'[ ]\d+[ ]+of[ ]+\d+','',text,flags=re.IGNORECASE)  #remove footer pattern where form number appears first
    text = re.sub(r'Page[ ]+\d+[ ]+of[ ]+\d+[ ]+©[ ]+Insurance'r'[ ]+Services[ ]+Office,[ ]+Inc\.,[ ]+\d{4}'r'[ ]+CA[ ]+[A-Z0-9 ]+','',text,flags=re.IGNORECASE)  #remove reverse footer pattern
    text = re.sub(r'[ \t]+',' ',text) #normalize spaces and tabs
    text = re.sub(r'\n\s*\n+','\n\n',text) #reduce excessive blank lines
    text = text.strip() #remove whitespace around the whole page
    return text

#---Phase 3: Metadata Enrichment Function---#

def enrich_metadata(doc):
    text = doc.page_content #doc-> langchain document; get text from the current doc
    text_lower = text.lower()
    top_text = text_lower[:800]
    
    #create default metadata values 
    document_type = "other"
    form_number = None
    section = None
    endorsement_name = None

#---Extract Form Number---#

    form_match = re.search(r'\bCA(?:\s+[A-Z]{1,3})?(?:\s+\d{2}){3,4}\b',text)

    if form_match:
        form_number = form_match.group(0)

#---Detect Document Type---#
    
    if "business auto declarations" in top_text:
        document_type = "declarations"
    
    elif "schedule of coverages and covered autos" in top_text:
        document_type = " coverage_schedule"

    elif "schedule of covered autos you own" in top_text: 
        document_type = "vehicle_schedule"

    elif "this endorsement changes the policy"  in top_text:
        document_type = "endorsement"
    
    elif "business auto coverage form" in top_text: 
        document_type = "coverage_form"



#---Detect Major Section Headings---#
    
    if ("section ii" in text_lower and "covered autos liability" in text_lower): 
        section = ("SECTION II - COVERED AUTOS LIABILITY COVERAGE")
    
    elif("section i" in text_lower and "covered autos" in text_lower):
        section = "SECTION I - COVERED AUTOS"
    
    elif "item one" in text_lower: 
        section = "ITEM ONE"
    
    elif "item two" in text_lower: 
        section = "ITEM TWO"
    
    elif "item three" in text_lower: 
        section = "ITEM THREE"

#---Dectect Endorsement Name---#
    
    if document_type == "endorsement":

        if "auto medical payments coverage" in text_lower:
            endorsement_name = ("AUTO MEDICAL PAYMENTS COVERAGE")

        elif "individual named insured" in text_lower: 
            endorsement_name = ("INDIVIDUAL NAMED INSURED")

#---Add metadata to existing metadata dict---#

    doc.metadata["document_type"] = document_type
    doc.metadata["form_number"] = form_number
    doc.metadata["section"] = section
    doc.metadata["endorsement_name"] = endorsement_name

    return doc

#---Phase 4: Process the entire corpus---#

#--- till now documents = RAW PyPDFLoader Documents; now processed_documents = cleaned + metadata-enriched documents---#

processed_documents = []

for doc in documents:  #loop through all 124 pages
    doc.page_content = clean_text(doc.page_content) #rawpage -> clean_text() -> cleaned page_content
    doc = enrich_metadata(doc) #adds: document_type, form_number, section, endorsement_name
    processed_documents.append(doc)

print("\nTotal processed documents:",len(processed_documents))  #to verify that no lose pages

#---Phase 5: Logical Segmentation / Metadata Inheritance---#

def segment_documents(processed_documents):

#Variables act like memory while move through PDF from page to page.
    current_form_number = None  #form currently inside
    current_parent_document_type = None #larger document type
    current_logical_document_id = None #Id shared by all pages belonging to the same logical form
    logical_document_counter = 0 #used to create : logical doc 1, logical doc 2,...

    segmented_documents = [] #final output

    #loop through all already processed pages in order

    for doc in processed_documents:
        detected_form_number = doc.metadata.get("form_number")
        detected_document_type = doc.metadata.get("document_type")

        if(detected_form_number is not None and detected_form_number != current_form_number):
            logical_document_counter += 1
            current_logical_document_id = (f"logical_doc_{logical_document_counter}")
            current_form_number = detected_form_number
            current_parent_document_type = detected_document_type
        
        if detected_form_number is None:
            doc.metadata["form_number"] = current_form_number
        
        doc.metadata["parent_document_type"] = (current_parent_document_type)
        doc.metadata["logical_document_id"] = (current_logical_document_id)

        segmented_documents.append(doc)

    return segmented_documents

segmented_documents = segment_documents(processed_documents)
print("Total segmented documents:", len(segmented_documents))





