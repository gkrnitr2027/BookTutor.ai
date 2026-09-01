# 📚 BookTutor AI

An AI-powered learning assistant that transforms PDF books into an interactive tutor. Using Retrieval-Augmented Generation (RAG), the application enables users to ask questions about uploaded books and receive accurate, context-aware answers. It is designed to make studying more interactive by combining document retrieval with modern Large Language Models (LLMs).

---

## Features

- 📖 Upload any PDF book or study material
- 🤖 AI-powered question answering using RAG
- 🔍 Semantic search for relevant book content
- 💬 Interactive chat interface
- ⚡ Fast retrieval with locally stored embeddings
- 💾 Saves processed books for faster future queries
- 🔒 Supports local LLM deployment for privacy

---

## Tech Stack

### Backend
- Python
- LangChain
- FAISS / ChromaDB
- PyMuPDF
- Sentence Transformers

### AI Models
- Local LLM (LM Studio / Ollama)
- Hugging Face Embedding Models

### Libraries
- NumPy
- Pandas
- Scikit-learn

---

## Project Structure

```
BookTutor-AI/
│
├── sample pdf
├── embeddings/
├── booktutor.py
├── requirements.txt
├── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gkrnitr2027/BookTutor.ai.git
cd BookTutor-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Start a local LLM server using LM Studio or Ollama.

Configure the endpoint inside the project if needed.

---

## Usage

Run the application with any PDF:

```bash
python booktutor.py path/to/book.pdf
```

The application will:

1. Read the PDF document
2. Split the content into searchable chunks
3. Generate vector embeddings
4. Build a searchable knowledge base
5. Launch an interactive AI tutoring session

---

## Workflow

```
PDF Book
    │
    ▼
Text Extraction
    │
    ▼
Chunking
    │
    ▼
Embedding Generation
    │
    ▼
Vector Database
    │
    ▼
Semantic Search
    │
    ▼
Large Language Model
    │
    ▼
AI Tutor Response
```

---

## Future Improvements

- Multi-book support
- Web-based interface
- Voice interaction
- Study notes generation
- Flashcard generation
- Quiz generation
- Citation-aware responses
- Multi-language support

---

## Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- LangChain
- Vector Databases
- Embedding Models
- Semantic Search
- PDF Processing
- Python Development
- AI Application Development

