import os
import logging
import nltk        
import os, uuid, time, logging, pathlib
from typing import List

from google.genai import types
from google import genai
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.utilities import SerpAPIWrapper
from langchain.prompts import PromptTemplate


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

google_client = genai.Client(api_key=GOOGLE_API_KEY)

if not all([GOOGLE_API_KEY, SERPAPI_API_KEY]):
    raise ValueError("One or more required Google environment variables are not set.")

_template = """
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

ANSWER_PROMPT = PromptTemplate.from_template(
    """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    Your answer must be extremely concise and direct.
    If the question asks "who", "what", or "where", and the answer is a name or a single term, provide only that name or term or more and more specific answer.
    Do not add any conversational filler, explanations, or introductory phrases like "According to the context...".
    Answer in the same language as the question.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

class RAGService:
    def __init__(self):
        logger.info("Initializing RAG Service...")
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("NLTK 'punkt' package not found. Downloading...")
            nltk.download('punkt')
            logger.info("NLTK 'punkt' downloaded successfully.")

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, google_api_key=GOOGLE_API_KEY)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        
        self.retrievers = {}
        
        self.web_search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
        logger.info("RAG Service initialized.")

    def process_and_upload_pdf(self, file_path: str, session_id: str):
        try:
            logger.info(f"Starting PDF processing for session {session_id} using file {file_path}")
            documents = process_without_prompt(file_path)

            text_splitter = SemanticChunker(
                self.embeddings, breakpoint_threshold_type="percentile"
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                raise ValueError("No text could be extracted or chunked from the PDF.")

            logger.info(f"Generated {len(chunks)} semantic chunks for session {session_id}.")
            
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            keyword_retriever = BM25Retriever.from_documents(chunks)
            
            self.retrievers[session_id] = {
                "vector_store": vector_store,
                "keyword_retriever": keyword_retriever,
                "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
            }

            logger.info(f"Successfully created and stored retrievers in memory for session {session_id}.")
            return {"message": f"File processed for session {session_id} with hybrid search and stored in memory."}
        except Exception as e:
            logger.error(f"Error in process_and_upload_pdf for session {session_id}: {e}")
            raise

    def answer_query(self, query: str, session_id: str):
        try:
            session_data = self.retrievers.get(session_id)
            if not session_data:
                logger.warning(f"No document loaded for session {session_id}. Falling back to web search.")
                return self._web_search_fallback(query)

            vector_store = session_data["vector_store"]
            keyword_retriever = session_data["keyword_retriever"]
            memory = session_data["memory"]
            
            semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            mq_retriever = MultiQueryRetriever.from_llm(retriever=semantic_retriever, llm=self.llm)
            embeddings_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.75)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter, base_retriever=mq_retriever
            )
            keyword_retriever.k = 10
            ensemble_retriever = EnsembleRetriever(
                retrievers=[compression_retriever, keyword_retriever],
                weights=[0.7, 0.3]
            )
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=ensemble_retriever,
                memory=memory,
                condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT},
                return_source_documents=True,
                chain_type="stuff",
            )
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "").strip()

            fallback_triggers = ["i don't know", "i cannot answer", "i do not have enough information"]
            if any(trigger in answer.lower() for trigger in fallback_triggers) or not answer:
                logger.warning(f"Answer indicates lack of info: '{answer}'. Falling back to web search.")
                return self._web_search_fallback(query)
                
            return {
                "answer": answer,
                "source": "Document (Hybrid Search)",
                "source_documents": [doc.page_content for doc in result.get("source_documents", [])]
            }

        except Exception as e:
            logger.error(f"Error in answer_query for session {session_id}: {e}")
            raise
    
    def _web_search_fallback(self, query: str) -> dict:
        logger.info(f"Executing web search for query: '{query}'")
        search_results = self.web_search.run(query)
        
        if not search_results or "Could not find any results for" in search_results:
            return {"answer": "I could not find an answer in the loaded document or on the web.", "source": "None", "source_documents": []}

        prompt = f"""
            Based on the following web search results, provide a direct and concise answer to the user's question.
            If the question asks for a specific name, place, or term, answer with only that.
            Do not use any introductory phrases. Answer in the same language as the question.

            Question: "{query}"

            Search Results:
            "{search_results}"

            Direct Answer:
        """
        
        response = self.llm.invoke(prompt)
        return {
            "answer": response.content.strip(),
            "source": "Web Search",
            "source_documents": [search_results]
        }


def split_pdf(pdf_path, chunk_size: int = 3) -> list[tuple[str, int]]:
    output_dir = "pdfcontents"
    os.makedirs(output_dir, exist_ok=True) 

    output_files = []
    try:
        pdf = PdfReader(pdf_path)
        total_pages = len(pdf.pages)
        for start_page in range(0, total_pages, chunk_size):
            pdf_writer = PdfWriter()
            end_page = min(start_page + chunk_size, total_pages)
            for i in range(start_page, end_page):
                pdf_writer.add_page(pdf.pages[i])
                
            unique_id = str(uuid.uuid4())
            output_filename = os.path.join(output_dir, f"{unique_id}_chunk_p{start_page + 1}-p{end_page}.pdf")
            with open(output_filename, "wb") as out:
                pdf_writer.write(out)
            output_files.append((output_filename, start_page))
    except Exception as e:
        print(f"Error splitting PDF: {e}")
    return output_files


def upload_file_to_client(file_path):

    file_path = file_path[0]
    file_path = pathlib.Path(file_path)
    
    logger.info(f"Google client version: {genai.__version__}")
    
    file_upload = google_client.files.upload(path=file_path)
    
    while file_upload.state == "PROCESSING":
        logger.info(f"Waiting for file to be processed.")
        time.sleep(10)
        file_upload = google_client.files.get(name=file_upload.name)

    if file_upload.state == "FAILED":
        error_msg = f"File processing failed with state: {file_upload.state}"
        logger.error(error_msg)

    logger.info(f"File processing complete")
        
    return file_upload


def get_file_processing_prompt():
    
    user_prompt = f"""
            Analyze the provided **PDF** file, extract all its content, and retain the original structure without omitting any elements present in the PDF file.

            ### **1. Text Structure**  
            - Maintain the original **headings** (e.g., `#`, `##`, `###` for different levels).  
            - Preserve paragraphs, bullet points (`-` or `*`), and numbered lists (`1.` `2.` `3.`).  
            - Keep **bold** and *italicized* text as it appears in the document or others.

            ### **2. Images**  
            - If the document contains images, for instead of each image position must insert [Category, Content, Summary] this format:
            **[Category: Single-word label such as "signature," "logo," etc.,  
            Content: Extracted text (if available) or more and more brief description of this image. Extracted Content of this Image, 
            Summary: Short summary of this image]**

            ### 3. Tables
            - Convert tables into **Markdown tables** while preserving their structure.
            - If a table is too complex (e.g., multi-column layouts), describe its format clearly.

            ### 4. Multi-Column Layouts
            - If the document contains multi-column sections, preserve the orginal structure.
            - Clearly indicate separations between sections.

            Do **not** include any introductory or explanatory remarks—only the extracted content should be provided.
            """
    
    return user_prompt


def process_file_with_gemini(user_prompt, file_upload):

    SYSTEM_PROMPT = f"""You are an assistant designed to read PDF files while preserving their **original structure**, including text, images, tables, and formatting. Your goal is to return an AI-generated version of the PDF that **mirrors** the original structure while remaining accessible and readable in Markdown.
            If the file contains images, each image **must** be included in its original position with the following format:
            [Category: Category of image within 2 words (e.g. Signature, Nature etc.), ImageName: <Image_PageNo_ImageIndex.Extension, e.g., Image_1_2.png>, Image Content: Image content as the image contains – do not remove or add anything]
            Do not alter or omit any part of the image content description. Maintain precise placement and context for each image."""
    
    
    try:        
        response = google_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=file_upload.uri,
                            mime_type=file_upload.mime_type),
                    ]),
                user_prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4
            ),
        )
        
        raw_result = response.text.replace("```markdown\n", "").replace("\n```", "")
        total_input_token_count = response.usage_metadata.prompt_token_count
        total_output_token_count = response.usage_metadata.candidates_token_count
        total_token_count = response.usage_metadata.total_token_count
        
        if raw_result.strip().startswith("```"):
            raw_result = raw_result.split("\n", 1)[1].strip()
        if raw_result.endswith("```"):
            raw_result = raw_result[:-3]
        
        return raw_result, total_input_token_count, total_output_token_count, total_token_count, "gemini-2.0-flash"
  

    except Exception as e:
            logger.error(f"Goes to the Exception Block...")
            raise Exception(f"{str(e)}")
        
        
def delete_file_from_client(file_upload):
    try:
        google_client.files.delete(name=file_upload.name)
        logger.info(f"File {file_upload.name} deleted successfully.")

    except Exception as e:
        if hasattr(e, 'status') and e.status == 'PERMISSION_DENIED':
            logger.error(f"Permission denied or file does not exist...")
            raise Exception(f"Permission denied or file does not exist...")
        else:
            logger.error(f"Failed to delete file...")
            raise Exception(f"Failed to delete file...")
        

def _process_single_segment(segment_info: tuple) -> tuple:
    idx, pdf_segment = segment_info
    segment_file_path = pdf_segment[0]
    
    try:
        logger.info(f"[Thread-{idx}] Starting processing for segment: {os.path.basename(segment_file_path)}")

        logger.info(f"[Thread-{idx}] Uploading to Gemini File API...")
        file_up = upload_file_to_client(pdf_segment)
        logger.info(f"[Thread-{idx}] Upload complete.")

        prompt = get_file_processing_prompt()

        logger.info(f"[Thread-{idx}] Extracting content with Gemini...")
        raw_result, input_token, output_token, token_count, model = process_file_with_gemini(prompt, file_up)
        logger.info(f"[Thread-{idx}] Content extraction complete.")
        
        document = Document(
            page_content=raw_result,
            metadata={
                "source": segment_file_path,
                "segment_index": idx,
                "segment_file": segment_file_path,
                "input_tokens": input_token,
                "output_tokens": output_token,
                "total_tokens": token_count,
                "model": model,
                "processing_method": "gemini_file_processing_concurrent"
            }
        )

        logger.info(f"[Thread-{idx}] Deleting remote file from Gemini...")
        delete_file_from_client(file_up)

        return idx, document

    except Exception as e:
        logger.error(f"[Thread-{idx}] Error processing segment {segment_file_path}: {e}")
        raise
    finally:
        try:
            os.remove(segment_file_path)
            logger.info(f"[Thread-{idx}] Removed local segment file: {os.path.basename(segment_file_path)}")
        except OSError as e:
            logger.error(f"[Thread-{idx}] Error removing local segment file {segment_file_path}: {e}")


def process_without_prompt(temp_file_path) -> List[Document]:
    output_dir = "pdfcontents"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True) 

    try:
        split_pdf_list = split_pdf(temp_file_path)
        if not split_pdf_list:
            logger.warning("PDF splitting resulted in no segments.")
            return []
            
        logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logger.info(f"Starting concurrent processing of {len(split_pdf_list)} PDF segments...")
        logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        results_with_indices = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_segment = {
                executor.submit(_process_single_segment, (idx, segment)): (idx, segment)
                for idx, segment in enumerate(split_pdf_list)
            }

            for future in as_completed(future_to_segment):
                segment_info = future_to_segment[future]
                try:
                    result = future.result()
                    results_with_indices.append(result)
                    logger.info(f"Successfully completed processing for segment {segment_info[0]}")
                except Exception as exc:
                    logger.error(f"Segment {segment_info[0]} ({os.path.basename(segment_info[1][0])}) generated an exception: {exc}")
                    raise Exception(f"Failed to process segment {segment_info[0]}. Aborting.")

        results_with_indices.sort(key=lambda x: x[0])
        documents = [doc for idx, doc in results_with_indices]

        if not documents:
            logger.error("All segments failed to process.")
            return []

        combined_input_token_count = sum(doc.metadata.get('input_tokens', 0) for doc in documents)
        combined_output_token_count = sum(doc.metadata.get('output_tokens', 0) for doc in documents)
        combined_token_count = sum(doc.metadata.get('total_tokens', 0) for doc in documents)

        documents[0].metadata.update({
            "source": temp_file_path,
            "total_input_tokens": combined_input_token_count,
            "total_output_tokens": combined_output_token_count,
            "total_combined_tokens": combined_token_count,
            "total_segments": len(documents)
        })

        logger.info(f"Successfully processed {len(documents)} document segments in total.")
        return documents

    except Exception as e:
        logger.error(f"An error occurred during the concurrent PDF processing workflow: {e}")
        raise
    
    finally:
        if os.path.exists(output_dir) and not os.listdir(output_dir):
            os.rmdir(output_dir)
            logger.info(f"Removed empty directory: {output_dir}")