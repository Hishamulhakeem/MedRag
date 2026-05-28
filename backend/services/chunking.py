import re
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata

def chunk_document(cleaned_text: str, report_id: str, filename: str) -> List[DocumentChunk]:
    """
    Splits text into chunks of 512 characters with 64 character overlap,
    tracking page numbers (parsing '--- Page N ---' separators) and assigning indexes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len
    )

    chunks: List[DocumentChunk] = []
    chunk_index = 0

    # Match page markers e.g. "--- Page 1 ---"
    page_splits = re.split(r'--- Page (\d+) ---', cleaned_text)
    
    if len(page_splits) > 1:
        # The structure is: [preamble, "1", "page 1 text", "2", "page 2 text", ...]
        # Preamble (if any) is assigned to page 1
        preamble = page_splits[0].strip()
        if preamble:
            sub_chunks = splitter.split_text(preamble)
            for ch in sub_chunks:
                chunks.append(DocumentChunk(
                    text=ch,
                    metadata={
                        "source_file": filename,
                        "chunk_index": chunk_index,
                        "page_number": 1,
                        "report_id": report_id
                    }
                ))
                chunk_index += 1

        for i in range(1, len(page_splits), 2):
            try:
                page_num = int(page_splits[i])
            except ValueError:
                page_num = 1
            
            page_text = page_splits[i+1].strip()
            if not page_text:
                continue

            sub_chunks = splitter.split_text(page_text)
            for ch in sub_chunks:
                chunks.append(DocumentChunk(
                    text=ch,
                    metadata={
                        "source_file": filename,
                        "chunk_index": chunk_index,
                        "page_number": page_num,
                        "report_id": report_id
                    }
                ))
                chunk_index += 1
    else:
        # No page indicators found, treat whole text as Page 1
        sub_chunks = splitter.split_text(cleaned_text)
        for ch in sub_chunks:
            chunks.append(DocumentChunk(
                text=ch,
                metadata={
                    "source_file": filename,
                    "chunk_index": chunk_index,
                    "page_number": 1,
                    "report_id": report_id
                }
            ))
            chunk_index += 1

    return chunks
