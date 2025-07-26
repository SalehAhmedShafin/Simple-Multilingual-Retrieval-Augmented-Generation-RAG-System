This project implements a sophisticated, multilingual Retrieval-Augmented Generation (RAG) system using FastAPI. It is designed to process PDF documents, answer questions about their content in multiple languages (e.g., English and Bengali), and maintain a conversational context.

The system features an advanced retrieval pipeline that combines semantic and keyword search for high relevance. It also includes a robust fallback mechanism that performs a web search if the answer cannot be found in the provided document.

## Key Features

-   **Advanced PDF Ingestion**: Uses Google's Gemini model (API) to extract structured text, tables, and image descriptions from PDFs, preserving the original layout.
-   **Hybrid Search**: Combines dense vector search (FAISS) for semantic understanding and sparse keyword search (BM25) for precise term matching using an `EnsembleRetriever`.
-   **State-of-the-Art Retrieval**: Enhances retrieval with `MultiQueryRetriever` to overcome vocabulary mismatch and `ContextualCompressionRetriever` to filter for the most relevant results.
-   **Conversational Memory**: Maintains session-specific chat history, allowing for contextual follow-up questions.
-   **Web Search Fallback**: Automatically queries the web via SerpApi if the document does not contain the necessary information, ensuring a comprehensive answer.
-   **Session-Based Interaction**: Manages document context and conversation history through unique session IDs.
-   **Async API**: Built with FastAPI for high-performance, asynchronous request handling.

## Setup Guide

### 1. Prerequisites

-   Python 3.9+
-   Docker (optional, for containerized deployment)

### 2. Local Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    A `requirements.txt` file would be ideal. Based on the imports, install the following:
    ```bash
    pip install -r requirements.txt
    ```
    or
    
    ```bash
    pip install "fastapi[all]" langchain langchain-google-genai google-generativeai pypdf2 faiss-cpu rank_bm25 nltk python-dotenv langchain-experimental
    ```

4.  **Download NLTK Data**
    The service automatically downloads the 'punkt' tokenizer on first run, but you can do it manually:
    ```python
    import nltk
    nltk.download('punkt')
    ```

5.  **Set Up Environment Variables**
    Create a file named `.env` in the root directory and add your API keys:
    ```.env
    GOOGLE_API_KEY="YOUR_GOOGLE_AI_STUDIO_API_KEY"
    SERPAPI_API_KEY="YOUR_SERPAPI_API_KEY"
    ```

6.  **Run the Application**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    The API will be available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.

### 3. Docker Deployment

1.  **Build the Docker Image**
    ```bash
    docker build -t multilingual-rag-app .
    ```

2.  **Run the Docker Container**
    Replace the placeholder values with your actual API keys.
    ```bash
    docker run -p 8000:8000 \
      -e GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY" \
      -e SERPAPI_API_KEY="YOUR_SERPAPI_API_KEY" \
      multilingual-rag-app
    ```

## Used Tools, Libraries, and Packages

| Category                  | Tool / Library                                                                               | Purpose                                                                                |
| ------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **Web Framework**         | `FastAPI`                                                                                    | Building the high-performance, asynchronous API.                                       |
| **LLM & Embeddings**      | `google-generativeai`, `langchain-google-genai`                                              | Interacting with Gemini Pro models (`gemini-2.5-pro`, `gemini-2.0-flash`, `text-embedding-004`). |
| **RAG Framework**         | `LangChain`                                                                                  | Orchestrating the entire RAG pipeline.                                                 |
| **Text Extraction**       | `Gemini 2.0 Flash` (via File API)                                                            | Intelligent extraction of structured content from PDFs.                                |
| **Vector Store**          | `FAISS`                                                                                      | Efficient similarity search on text embeddings (semantic search).                      |
| **Keyword Search**        | `rank-bm25`                                                                                  | BM25 algorithm for keyword-based retrieval.                                            |
| **Core Components**       | `PyPDF2`, `NLTK`, `python-dotenv`                                                            | PDF splitting, text tokenization, and environment variable management.                 |
| **Web Search**            | `SerpApi`                                                                                    | Providing real-time web search results as a fallback mechanism.                        |

## Sample Queries and Outputs

Let's assume we uploaded a PDF biography of Sheikh Mujibur Rahman.

#### 1. English Query (Document Source)

**Request:** `POST /query`
```json
{
  "query": "Who was the first president of Bangladesh?",
  "session_id": "2055ff91-f47a-492c-a73d-0a254ddb8ef3"
}
```

**Response:**
```json
{
  "answer": "Sheikh Mujibur Rahman",
  "source": "Document (Hybrid Search)",
  "source_documents": [
    "Following the Declaration of Independence in April 1971, Sheikh Mujibur Rahman was named as the first president of the provisional government of Bangladesh...",
    "..."
  ]
}
```

**Outside of Document Request:** `POST /query`
```json
{
  "query": "Who won the 2024 Super Bowl?",
  "session_id": "1155ff91-f47a-492c-a73d-0a254ddb8e90"
}
```
**Response:**
```json
{
  "answer": "Kansas City Chiefs",
  "source": "Web Search",
  "source_documents": [
    "The Kansas City Chiefs defeated the San Francisco 49ers 25-22 in overtime in Super Bowl LVIII on Sunday in Las Vegas. Patrick Mahomes was named the MVP..."
    "..."
  ]
}
```

#### 2. Bengali Query (Document Source)

**Request:** `POST /query`
```json
{
  "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
  "session_id": "2055ff91-f47a-492c-a73d-0a254ddb8yuqq"
}
```

**Response:**
```json
{
  "answer": "১৫ বছর",
  "source": "Document (Hybrid Search)",
  "source_documents": [
    "১৯৭১ সালের এপ্রিল মাসে স্বাধীনতার ঘোষণার পর শেখ মুজিবুর রহমানকে বাংলাদেশের প্রথম রাষ্ট্রপতি হিসেবে নামকরণ করা হয়...",
    "..."
  ]
}
```

**Outside of Document Request:** `POST /query`
```json
{
  "query": "বাংলাদেশের প্রথম রাষ্ট্রপতি কে ছিলেন?",
  "session_id": "9055ff91-f47a-492c-a73d-0a254ddb8ef3"
}
```

**Response:**
```json
{
  "answer": "শেখ মুজিবুর রহমান",
  "source": "Web Search",
  "source_documents": [
    "১৯৭১ সালের এপ্রিল মাসে স্বাধীনতার ঘোষণার পর শেখ মুজিবুর রহমানকে বাংলাদেশের প্রথম রাষ্ট্রপতি হিসেবে নামকরণ করা হয়...",
    "..."
  ]
}
```

#### 3. English Query (Web Search Fallback)

This query is unrelated to the PDF content, triggering the fallback.

**Request:** `POST /query`
```json
{
  "query": "What is the current price of gold?",
  "session_id": "2055ff91-f47a-492c-a73d-0a254ddb8eg5"
}
```
**Response:**
```json
{
  "answer": "As of today, the price of gold is approximately $2,350 per ounce.",
  "source": "Web Search",
  "source_documents": [
    "Summary of web search results from SerpApi..."
  ]
}
```

## API Documentation

The API is self-documenting via Swagger UI and ReDoc, available at `/docs` and `/redoc` respectively.

### `POST /upload-pdf`

Processes and ingests a PDF document for querying. This creates a session for the document.

-   **Request Body**: `multipart/form-data`
    -   `session_id` (string, optional): A unique ID for the session. If not provided, a new one is generated.
    -   `file` (file, required): The PDF file to be processed.
-   **Success Response** (`200 OK`):
    ```json
    {
      "message": "File processed for session {session_id} with hybrid search and stored in memory.",
      "session_id": "generated-or-provided-session-id"
    }
    ```

### `POST /query`

Submits a query against a previously uploaded document or the web.

-   **Request Body**: `application/json`
    ```json
    {
      "query": "Your question here",
      "session_id": "The session ID returned by /upload-pdf"
    }
    ```
-   **Success Response** (`200 OK`):
    ```json
    {
      "answer": "The generated answer.",
      "source": "Document (Hybrid Search) | Web Search | None",
      "source_documents": ["List of source text chunks used to generate the answer."]
    }
    ```

## Evaluation Matrix

Formal, automated evaluation is not yet implemented in this codebase but is a critical next step for productionizing a RAG system. An evaluation pipeline would typically measure the following:

-   **Answer Relevancy**: How relevant is the answer to the user's query?
-   **Faithfulness**: Does the answer stay true to the provided context? (i.e., does it avoid making things up?)
-   **Context Precision**: Are the retrieved chunks relevant to the query?
-   **Context Recall**: Were all relevant chunks successfully retrieved from the document?

**Future Implementation**: Frameworks like **Ragas** or **LangChain's evaluation tools** could be integrated to create a test set of question-answer pairs and automatically score the system's performance on these metrics.

## Answer Following Questions

#### 1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

-   **Method**: The application uses a highly advanced method for text extraction. Instead of a traditional library like `PyPDF2` or `pdfplumber` for raw text dumps, it leverages the **Google Gemini model (`gemini-2.0-flash`) via its File API**. The PDF is split into smaller chunks (3 pages each), and each chunk is uploaded and processed by Gemini.
-   **Why this Method was Chosen**: This approach was chosen because it performs **intelligent content extraction**, not just text scraping. The prompt given to Gemini instructs it to:
    -   Preserve the original document structure (headings, lists, paragraphs).
    -   Convert tables into Markdown format.
    -   Analyze images and provide a structured description: `[Category, Content, Summary]`.
    This provides a much richer, more structured, and semantically meaningful representation of the content than a simple text dump, which is crucial for high-quality retrieval.
-   **Formatting Challenges Addressed**: This method directly addresses common PDF formatting challenges. Simple libraries often struggle with multi-column layouts, complex tables, and embedded images. By using a powerful multimodal LLM to "read" the PDF, these challenges are effectively solved, as the model can interpret the visual layout and structure of the page.

#### 2. What chunking strategy did you choose? Why do you think it works well for semantic retrieval?

-   **Chunking Strategy**: The application uses `langchain_experimental.text_splitter.SemanticChunker`.
-   **Why it Works Well**: This is a state-of-the-art chunking strategy. Unlike fixed-size or paragraph-based chunking, `SemanticChunker` splits the text based on **semantic meaning**. It calculates the embedding (the vector representation) of consecutive sentences and introduces a split when the semantic distance between them exceeds a certain threshold. This is superior for semantic retrieval because:
    1.  **Contextual Cohesion**: It keeps sentences that are thematically related together in the same chunk.
    2.  **Meaningful Boundaries**: Chunks are not broken awkwardly in the middle of a thought or argument.
    3.  **Improved Retrieval**: When a query is made, the retrieved chunks are more likely to contain a complete, self-contained piece of relevant information, leading to better-quality answers from the LLM.

#### 3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

-   **Embedding Model**: `models/text-embedding-004` from Google.
-   **Why it was Chosen**: This is one of Google's latest and most powerful text embedding models. It was chosen for its:
    -   **High Performance**: Designed specifically for state-of-the-art performance in retrieval tasks (RAG).
    -   **Efficiency**: Offers strong performance while being computationally efficient.
    -   **Ecosystem Synergy**: It comes from the same family as the Gemini models used for generation, which can lead to better alignment between how text is understood (embedded) and how it is processed (generated).
-   **How it Captures Meaning**: The model converts any piece of text into a high-dimensional numerical vector. It is trained on a massive dataset, learning to place texts with similar meanings close to each other in this vector space. For example, the vectors for "monarchy's ruler" and "the king of the nation" would be very close. This allows the system to find relevant chunks even if they don't use the exact keywords from the query.

#### 4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

-   **Comparison Method**: A sophisticated hybrid approach using LangChain's `EnsembleRetriever`. It combines two distinct methods:
    1.  **Semantic Similarity (FAISS)**: The user's query is converted into a vector using the same embedding model. **FAISS** (Facebook AI Similarity Search) is then used to perform a very fast nearest-neighbor search to find the chunks with the closest vectors (i.e., the most semantically similar chunks).
    2.  **Keyword Matching (BM25)**: The `BM25Retriever` uses a classic information retrieval algorithm (Okapi BM25) to find chunks that contain the specific keywords from the query. It's excellent for finding literal matches like names, acronyms, or specific terms.
-   **Why this Setup was Chosen**: This hybrid setup is the "best of both worlds". Semantic search is great for understanding the *intent* and *meaning* behind a query, but can sometimes miss specific, literal terms. Keyword search is perfect for those literal terms but understands no context or synonyms. By combining them (`weights=[0.7, 0.3]` favoring semantic), the retriever is robust, accurate, and less likely to miss relevant information. **FAISS** was chosen as the vector store because it is highly optimized, memory-efficient, and industry-standard for fast similarity searches.

#### 5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

-   **Ensuring Meaningful Comparison**: The system uses two key techniques:
    1.  **Multi-Query Retriever**: The semantic search part is wrapped in a `MultiQueryRetriever`. This uses an LLM (`gemini-2.5-pro`) to take the user's original query and generate several different variations of it from different perspectives. For a query like "king's wealth," it might generate "What were the king's assets?" and "How rich was the monarch?". The system then retrieves documents relevant to *all* of these queries, broadening the search and making it more resilient to the user's specific phrasing.
    2.  **Conversational Chain**: The `ConversationalRetrievalChain` with `ConversationBufferMemory` handles conversational context. If a user asks a follow-up question like "What about his son?", the chain first uses the chat history to rephrase it into a standalone question, e.g., "What about Sheikh Mujibur Rahman's son?". This ensures the rephrased, context-rich question is compared to the document chunks.

-   **Handling Vague Queries**: If a query is vague and retrieval fails to find relevant information, the system has a powerful **fallback mechanism**. The LLM will likely generate an answer like "I don't know" or "I cannot answer based on the provided context." The code explicitly checks for these trigger phrases. If detected, it automatically initiates a **web search** using `SerpAPIWrapper`. The results of this web search are then used to generate a final answer, ensuring the user almost always gets a helpful response.

#### 6. Do the results seem relevant? If not, what might improve them?

-   **Relevance**: Given the architecture, the results are expected to be **highly relevant**. The combination of intelligent text extraction (Gemini), semantic chunking, and a hybrid multi-query retrieval pipeline is a state-of-the-art approach designed to maximize relevance.
-   **Potential Improvements**:
    1.  **Re-ranking**: After the initial retrieval of (e.g.,) 10 chunks by the `EnsembleRetriever`, a more powerful model could be used to re-rank these 10 chunks for relevance before passing the top 3-5 to the final LLM for answer synthesis. This adds a quality-control step.
    2.  **Fine-Tuning**: For a highly specialized domain (e.g., legal or medical documents), the embedding model (`text-embedding-004`) could be fine-tuned on a corpus of domain-specific text to better understand its unique vocabulary and nuances.
    3.  **Dynamic Weighting**: The weights for the `EnsembleRetriever` are currently fixed (`[0.7, 0.3]`). An advanced strategy could dynamically adjust these weights based on the query's nature (e.g., give more weight to keyword search if the query contains many proper nouns or codes).
    4.  **Chunking Strategy Experimentation**: While `SemanticChunker` is excellent, for certain document types, experimenting with different chunk sizes or using a `RecursiveCharacterTextSplitter` with chunk overlap might yield better results.
    5.  **Chain Type for Very Large Context**: The current implementation uses the `stuff` chain type, which puts all retrieved context into a single prompt. If the total context from a very large document exceeds the LLM's context window, switching to a more scalable chain type like `Map-Reduce` or `Refine` would be necessary.
