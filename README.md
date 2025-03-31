# PDF RAG System

A Retrieval Augmented Generation (RAG) system for PDF documents using LangChain, ChromaDB, and OpenAI or Ollama.

## Overview

This project implements a RAG system that can:

1. Ingest PDF documents and store them in a vector database (ChromaDB)
2. Process and chunk documents using PyMuPDF and LangChain
3. Generate embeddings using OpenAI's embedding models or local Ollama models
4. Retrieve relevant context using similarity search
5. Generate answers to questions using LLMs (OpenAI or local models via Ollama)
6. Expose functionality via a FastAPI web service

## LLM Provider Options

The system supports two LLM providers:

1. **OpenAI**: Uses OpenAI's API for embeddings and completions (requires an API key)
2. **Ollama**: Uses [Ollama](https://ollama.ai/) for local, open-source LLMs (no API key required)

You can choose the provider by setting the `LLM_PROVIDER` environment variable in the `.env` file.

## Project Structure

```
Rag/
├── src/                        # Source code
│   ├── api/                    # API layer
│   │   ├── app.py              # FastAPI application
│   │   ├── routes.py           # API routes
│   │   └── schemas.py          # API schemas using Pydantic
│   ├── config/                 # Configuration
│   │   └── config.py           # Configuration settings
│   ├── database/               # Database layer
│   │   └── chroma_db.py        # ChromaDB connector
│   ├── ingest/                 # Document ingestion
│   │   ├── document_processor.py # Document processing
│   │   └── ingest_documents.py # Document ingestion scripts
│   ├── models/                 # RAG models
│   │   └── rag_model.py        # RAG implementation
│   └── utils/                  # Utilities
├── data/                       # Data directory
│   └── pdfs/                   # PDF documents
├── chroma_db/                  # ChromaDB storage
├── tests/                      # Unit tests
├── main.py                     # Main entry point
├── Makefile                    # Automation for setup and operations
├── setup.bat                   # Windows setup script
├── setup.sh                    # Unix/macOS setup script
├── switch_provider.py          # Script to switch between LLM providers
├── requirements.txt            # Dependencies
├── .env                        # Environment variables
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8+ installed on your system
- For Makefile usage: `make` utility (see below for installation instructions)
- For Ollama usage: [Ollama](https://ollama.ai/) installed and running locally

### Option 1: Using Makefile (Recommended)

The project includes a cross-platform Makefile for easy setup and operation. It automatically detects if you're using Windows, macOS, or Linux and adapts accordingly.

1. Clone the repository
2. Set up your environment using the Makefile:
   ```
   make setup
   ```
   This will create a virtual environment and install all dependencies.

   Running `make help` will show the detected operating system and available commands.

#### Make Installation

- **Windows**: Install [Make for Windows](https://gnuwin32.sourceforge.net/packages/make.htm) or use [Chocolatey](https://chocolatey.org/): `choco install make`
- **macOS**: Comes pre-installed or install with Homebrew: `brew install make`
- **Linux**: Install using package manager: `sudo apt-get install make` (Ubuntu/Debian) or `sudo yum install make` (CentOS/RHEL)

### Option 2: Using Setup Scripts

If you don't have Make installed, you can use the provided setup scripts:

#### Windows
Run the batch file by double-clicking it or from Command Prompt:
```
setup.bat
```

#### macOS/Linux
Run the shell script:
```
./setup.sh
```
If you get a permission error, make it executable first:
```
chmod +x setup.sh
```

### Option 3: Manual Setup

If you prefer to set up manually:

1. Clone the repository
2. Create a virtual environment:
   ```
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Environment Variables

After setup, you need to configure environment variables by creating a `.env` file:
```
# Choose provider: "openai" or "ollama"
LLM_PROVIDER=openai

# OpenAI configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_COMPLETION_MODEL=gpt-3.5-turbo

# Ollama configuration (for local LLMs)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_COMPLETION_MODEL=llama2

# Storage paths
CHROMA_PERSIST_DIRECTORY=./chroma_db
PDF_DIRECTORY=./data/pdfs
```

### Using Ollama (Local Models)

To use Ollama instead of OpenAI:

1. [Install Ollama](https://ollama.ai/download) on your machine
2. Download the models you want to use:
   ```
   ollama pull llama2          # For completions
   ollama pull nomic-embed-text  # For embeddings
   ```
3. Make sure Ollama is running:
   ```
   ollama serve
   ```
4. Set `LLM_PROVIDER=ollama` in your `.env` file
5. Customize the model names in `.env` if needed

## Usage

### Starting the API Server

#### Using Makefile:
```
make run
```

#### Using direct scripts:
```
# On Windows
venv\Scripts\activate.bat
python main.py

# On macOS/Linux
source venv/bin/activate
python main.py
```

By default, the server runs on `http://0.0.0.0:8000`. You can access the API documentation at `http://localhost:8000/docs`.

### Ingesting Documents

1. Place your PDF documents in the `data/pdfs` directory.

2. Use the API endpoint to ingest a single PDF document:
   ```
   POST /api/v1/ingest
   {
     "file_path": "/path/to/document.pdf"
   }
   ```

3. Or use the API endpoint to ingest all PDFs in a directory:
   ```
   POST /api/v1/ingest/directory?directory_path=/path/to/pdf/directory
   ```

### Querying the RAG System

Use the query endpoint to ask questions about your documents:

```
POST /api/v1/query
{
  "query": "What does the document say about X?",
  "include_sources": true
}
```

Set `include_sources` to `true` to include source document information in the response.

## Switching Between LLM Providers

### Using Makefile (Easiest)

If you're using the Makefile, you can easily switch providers with:

```
# Switch to OpenAI
make use-openai

# Switch to Ollama
make use-ollama
```

These commands use the switch_provider.py script and handle all configuration automatically.

### Using the Provider Switch Script

You can also directly use the helper script:

```
# Switch to OpenAI
python switch_provider.py openai

# Switch to Ollama
python switch_provider.py ollama
```

The script automatically updates your `.env` file and provides helpful prompts for additional setup steps.

### Manual Configuration

You can also manually switch by changing the `LLM_PROVIDER` in your `.env` file:

```
# For OpenAI
LLM_PROVIDER=openai

# For local models using Ollama
LLM_PROVIDER=ollama
```

Remember that:
- OpenAI requires a valid API key
- Ollama requires the service to be running locally
- Different models may have different capabilities and performance characteristics

## Development

### Running Tests

Using Makefile:
```
make test
```

Or manually:
```
# On Windows
venv\Scripts\pytest tests/

# On macOS/Linux
source venv/bin/activate && pytest tests/
```

### Code Quality

Using Makefile:
```
make lint      # Run linter
make format    # Format code
```

### Cleaning Up

To clean up the virtual environment and cached files:
```
make clean
```

### Available Makefile Commands

Run `make help` to see all available commands:
```
make help
```

### Project Structure Explanation

- **api/**: Contains the FastAPI application, routes, and schema definitions.
- **config/**: Contains configuration settings loaded from environment variables.
- **database/**: Contains the ChromaDB connector for vector database operations.
- **ingest/**: Contains document processing and ingestion functionality.
- **models/**: Contains the RAG model implementation.

## License

This project is licensed under the MIT License.

## Acknowledgements

This project uses:
- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [OpenAI API](https://platform.openai.com/)
- [Ollama](https://ollama.ai/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyMuPDF](https://github.com/pymupdf/pymupdf) 