#!/bin/bash
# Clean and setup script for RAG Pipeline

echo "🧹 Cleaning and setting up RAG Pipeline environment..."

# Check if the venv exists
if [ -d "venv" ]; then
    echo "Found existing virtual environment, removing it..."
    rm -rf venv
    echo "✅ Removed old virtual environment"
fi

# Create a fresh virtual environment
echo "Creating new virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✅ Created and activated new virtual environment"

# Upgrade pip and install wheel
echo "Upgrading pip and installing wheel..."
pip install --upgrade pip
pip install wheel
echo "✅ Pip upgraded and wheel installed"

# Install the requirements with exact versions
echo "Installing dependencies with exact versions..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create necessary directories
mkdir -p data/pdfs
echo "✅ Created data directories"

# Create a simple .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating default .env file..."
    cat > .env << EOL
# Choose your LLM provider: "openai", "ollama", or "huggingface"
LLM_PROVIDER=ollama

# Choose your embedding provider (can be different from LLM provider)
EMBEDDING_PROVIDER=huggingface

# OpenAI configuration (if using OpenAI)
# OPENAI_API_KEY=your_openai_api_key
OPENAI_COMPLETION_MODEL=gpt-3.5-turbo

# Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=llama2
OLLAMA_COMPLETION_MODEL=llama2

# HuggingFace configuration
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_COMPLETION_MODEL=google/flan-t5-base

# Document and database settings
PDF_DIRECTORY=./data/pdfs
CHROMA_PERSIST_DIRECTORY=./chroma_db
EOL
    echo "✅ Created default .env file"
fi

# Create a run script
echo "Creating run script..."
cat > run_chatbot.sh << EOL
#!/bin/bash
# RAG Pipeline Chatbot Runner

# Activate the virtual environment
source venv/bin/activate

# Check for the --ingest flag
if [ "\$1" == "--ingest" ]; then
    echo "🔄 Running chatbot with document ingestion..."
    python src/rag_pipeline_chatbot.py --ingest
else
    echo "🔄 Running chatbot without document ingestion..."
    python src/rag_pipeline_chatbot.py
fi
EOL
chmod +x run_chatbot.sh
echo "✅ Created run script"

echo ""
echo "🎉 Setup complete! To run the chatbot:"
echo "  ./run_chatbot.sh         # Without document ingestion"
echo "  ./run_chatbot.sh --ingest  # With document ingestion"
echo ""
echo "📝 Make sure to:"
echo "  1. Configure your .env file with the desired settings"
echo "  2. Add PDF documents to the data/pdfs directory"
echo "  3. If using Ollama, ensure it's running with: ollama serve"
echo "     and pull the necessary model with: ollama pull llama2"
echo "" 