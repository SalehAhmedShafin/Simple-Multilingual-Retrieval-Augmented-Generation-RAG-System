import uuid
import logging
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from typing import List
from app.services.rag_service import RAGService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

try:
    rag_service_instance = RAGService()
except Exception as e:
    logging.error(f"Failed to initialize RAGService: {e}")
    raise

class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(..., description="A unique identifier for the conversation session.")

class AnswerResponse(BaseModel):
    answer: str
    source: str
    source_documents: List[str]

class UploadResponse(BaseModel):
    message: str
    session_id: str


@router.post(
    "/upload-pdf", 
    response_model=UploadResponse, 
    tags=["Document Management"],
    description="""
    Processes and ingests a PDF document for querying.

    This endpoint is the starting point for any document-based conversation. 
    It performs several key actions:
    
    - **File Handling:** Securely accepts a PDF file upload.
    - **Session Management:** Creates a unique session for the document. You can provide your own `session_id` or let the API generate one for you. This ID is crucial for linking subsequent queries to this specific document.
    - **Indexing for Hybrid Search:** The PDF's text is extracted, split into semantic chunks, and indexed in two ways:
        1.  A **semantic vector store (FAISS)** for understanding the meaning of the text.
        2.  A **keyword-based index (BM25)** for efficient term matching.
    - **In-Memory Storage:** The generated indexes are stored in the server's memory, associated with the `session_id`.

    **Workflow:** To start a new conversation, call this endpoint first. Use the returned `session_id` in all subsequent calls to the `/query` endpoint for this document.

    **Note:** If you upload a new PDF using an existing `session_id`, the previous document's data for that session will be **replaced**.
    """
)
async def upload_pdf(
    session_id: str = Form(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Optional: A unique ID for the session. If not provided, a new one will be generated."
    ),
    file: UploadFile = File(..., description="The PDF file to be processed.")
):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are accepted.")
    
    task_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            task_file_path = tmp_file.name
        logger.info(f"File for session {session_id} saved to temporary path: {task_file_path}")
        
        result = rag_service_instance.process_and_upload_pdf(task_file_path, session_id)
        result['session_id'] = session_id
        return result

    except Exception as e:
        logger.error(f"Error during PDF upload for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")
    finally:
        if task_file_path and os.path.exists(task_file_path):
            os.remove(task_file_path)
            logger.info(f"Cleaned up temporary file: {task_file_path}")
        await file.close()

@router.post(
    "/query", 
    response_model=AnswerResponse, 
    tags=["Q&A"],
    description="""
    Submits a query to get an answer based on a previously uploaded document or the web.

    This endpoint uses the `session_id` to retrieve the correct document context and conversation history.

    **Retrieval Process:**
    - **Session-Based Retrieval:** Uses the `session_id` to locate the document's semantic and keyword indexes in memory.
    - **Advanced Hybrid Search:** Combines results from both semantic and keyword searches (`EnsembleRetriever`) to find the most relevant document chunks. It enhances semantic search by generating multiple variations of the input query (`MultiQueryRetriever`).
    - **Contextual Memory:** Maintains a conversation history for the session, allowing for follow-up questions.
    
    **Fallback Mechanism:**
    - **Automatic Web Search:** If the initial answer generated from the document context is weak or indicates a lack of information (e.g., contains "I don't know"), the system automatically performs a web search using SerpApi.
    - **LLM-Powered Synthesis:** The results of the web search are then passed to the LLM to synthesize a comprehensive answer.

    The `source` field in the response will indicate whether the answer came from the "Document (Hybrid Search)" or "Web Search".

    **Prerequisite:** You must have successfully called `/upload-pdf` and obtained a `session_id` before using this endpoint for document-based Q&A.
    """
)
async def query_rag(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")
    
    try:
        result = rag_service_instance.answer_query(request.query, request.session_id)
        return result
    except Exception as e:
        logger.error(f"Error during query processing for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while answering the query: {e}")