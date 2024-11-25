# RAG Task Application

## Overview
This is a Retrieval-Augmented Generation (RAG) application that processes PDF, Word, and Excel documents, embeds the extracted content, and enables querying using Pinecone's vector store. The application uses **Gemini embeddings** for robust vector representation and supports various document types.

## Features
- **Document Processing**: Reads and processes content from PDF, Word, and Excel files.
- **Vector Embedding**: Uses Google Gemini embeddings for high-quality document representation.
- **Vector Store Integration**: Stores embedded documents in Pinecone for efficient retrieval.
- **Query Interface**: Allows users to query documents using natural language questions.

---

## Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/aurimasruk/rag-application.git
   ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create Gemini and Pinecone API keys and ADC credentials:
    - [For Gemini](https://ai.google.dev/gemini-api/docs/api-key)
    - [For Pinecone](https://www.pinecone.io/)
    - [For Google Cloud Credentials (ADC)](https://cloud.google.com/docs/authentication/provide-credentials-adc)

4. Set environment variables in a .env file:
    ```makefile
    GEMINI_API_KEY=<Your Gemini API Key>
    PINECONE_API_KEY=<Your Pinecone API Key>
    GOOGLE_APPLICATION_CREDENTIALS=<Path to Google Credentials JSON>
    ```

## Usage

1. Start the application:
    ```bash
    python main.py
    ```
    The application includes a utility to generate test documents for validation.

2. Follow the prompt to enter a question. Examples:
    - "What are the warranties of Product A and B?"
    - "What features does Product A have?"
    - "Which product is suitable for outdoor use?"

3. The application will retrieve and display an answer based on embedded documents.


## Testing

### Running automated tests

Automated tests are included to verify the application's core functionalities:

```bash
python test_rag_task.py
```

## Additional Notes

- Ensure your API keys are valid and all dependencies are installed together with correctly configured environment variables.
- Logs are set to ``ERROR`` level by default and can be adjusted to ``DEBUG`` or ``INFO`` for troubleshooting.
