from typing import Iterator
import time
import os
import argparse

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LangChain imports
from langchain_core.documents import Document as LCDocument
from langchain_core.document_loaders import BaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)


# ============================================================
# DOCLING PDF LOADER
# ============================================================

class DoclingBookLoader(BaseLoader):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.AUTO,
        )

        pipeline_options = PdfPipelineOptions()

        pipeline_options.accelerator_options = accelerator_options

        # Enable OCR
        pipeline_options.do_ocr = True

        # Enable table structure extraction
        pipeline_options.do_table_structure = True

        pipeline_options.table_structure_options.do_cell_matching = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def lazy_load(self) -> Iterator[LCDocument]:
        print(f"\n📚 Processing book: {self.file_path}")

        process_start = time.time()

        docling_doc = self.converter.convert(
            self.file_path
        ).document

        process_time = time.time() - process_start

        print(
            f"✅ Book processed successfully in "
            f"{process_time:.2f} seconds"
        )

        print("🔄 Converting to markdown format...")

        convert_start = time.time()

        text = docling_doc.export_to_markdown()

        convert_time = time.time() - convert_start

        print(
            f"✅ Conversion complete in "
            f"{convert_time:.2f} seconds"
        )

        metadata = {
            "source": self.file_path,
            "format": "book",
            "process_time": process_time,
            "convert_time": convert_time,
        }

        yield LCDocument(
            page_content=text,
            metadata=metadata,
        )


# ============================================================
# CREATE BOOK QA SYSTEM
# ============================================================

def create_book_qa_system(pdf_path: str):

    total_start_time = time.time()

    print("\n🚀 Initializing Book QA System...")

    # --------------------------------------------------------
    # FAISS INDEX PATH
    # --------------------------------------------------------

    index_path = f"{pdf_path}_faiss_index"

    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    print("\n🔤 Initializing embedding model...")

    embedding_start = time.time()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_init_time = time.time() - embedding_start

    print(
        f"✅ Embedding model initialized in "
        f"{embedding_init_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # LOAD OR CREATE FAISS VECTOR STORE
    # --------------------------------------------------------

    if os.path.exists(index_path):

        print("\n📦 Loading existing vector store...")

        load_start = time.time()

        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        load_time = time.time() - load_start

        print(
            f"✅ Vector store loaded in "
            f"{load_time:.2f} seconds"
        )

    else:

        print(
            "\n💫 No existing index found. "
            "Creating new one..."
        )

        # ----------------------------------------------------
        # LOAD PDF WITH DOCLING
        # ----------------------------------------------------

        loader = DoclingBookLoader(pdf_path)

        documents = loader.load()

        # ----------------------------------------------------
        # SPLIT DOCUMENT
        # ----------------------------------------------------

        print("\n📄 Splitting document into chunks...")

        split_start = time.time()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

        splits = text_splitter.split_documents(
            documents
        )

        split_time = time.time() - split_start

        print(
            f"✅ Created {len(splits)} chunks in "
            f"{split_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # CREATE FAISS VECTOR STORE
        # ----------------------------------------------------

        print(
            "\n📦 Building vector store "
            "and creating embeddings..."
        )

        vectorstore_start = time.time()

        vectorstore = FAISS.from_documents(
            splits,
            embeddings,
        )

        vectorstore_time = (
            time.time() - vectorstore_start
        )

        print(
            f"✅ Vector store built in "
            f"{vectorstore_time:.2f} seconds"
        )

        # ----------------------------------------------------
        # SAVE FAISS INDEX
        # ----------------------------------------------------

        print(
            f"\n💾 Saving vector store to "
            f"{index_path}"
        )

        save_start = time.time()

        vectorstore.save_local(index_path)

        save_time = time.time() - save_start

        print(
            f"✅ Vector store saved in "
            f"{save_time:.2f} seconds"
        )

    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
        },
    )

    print("✅ Vector store ready")

    # --------------------------------------------------------
    # LOCAL LLM / LM STUDIO
    # --------------------------------------------------------

    print("\n🤖 Connecting to local language model...")

    llm = ChatOpenAI(
        model="local-model",
        base_url="http://localhost:1234/v1",
        api_key="not-needed",
        temperature=0,
    )

    print("✅ Connected to local language model")

    # ========================================================
    # HISTORY-AWARE RETRIEVER
    # ========================================================

    print("\n🔎 Creating conversational retriever...")

    contextualize_q_system_prompt = """
Given a chat history and the latest user question,
rewrite the question into a standalone question.

The standalone question must be understandable without
the chat history.

Do NOT answer the question.

Return only the rewritten question.
"""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                contextualize_q_system_prompt,
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )

    history_aware_retriever = (
        create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt,
        )
    )

    # ========================================================
    # QUESTION ANSWERING PROMPT
    # ========================================================

    print("🧠 Creating answer generation chain...")

    qa_system_prompt = """
You are a helpful assistant answering questions about
a book.

Use ONLY the provided context to answer the user's question.

Context:

{context}

Rules:

1. Answer accurately based on the provided context.
2. Be concise but informative.
3. If the answer is not contained in the context, say:
   "I couldn't find that information in the provided book."
4. Do not invent facts.
5. If the question asks about a person, event, concept,
   or topic, explain it clearly using the available context.
"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                qa_system_prompt,
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )

    # --------------------------------------------------------
    # DOCUMENT CHAIN
    # --------------------------------------------------------

    question_answer_chain = (
        create_stuff_documents_chain(
            llm,
            qa_prompt,
        )
    )

    # ========================================================
    # COMPLETE RAG CHAIN
    # ========================================================

    qa_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain,
    )

    total_time = time.time() - total_start_time

    print(
        f"\n✨ System ready! "
        f"Total setup took {total_time:.2f} seconds"
    )

    return qa_chain


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    print("\n" + "=" * 80)

    print("📚 RETRIEVED CONTEXT CHUNKS:")

    print("=" * 80)

    # New retrieval chain returns documents in "context"
    documents = result.get("context", [])

    if not documents:

        print("\n⚠️ No context documents were returned.")

    else:

        for i, doc in enumerate(documents, 1):

            print(f"\nCHUNK {i}:")

            print("-" * 40)

            print(doc.page_content)

            print("-" * 40)

    print("\n" + "=" * 80)

    print("🤖 LLM RESPONSE:")

    print("=" * 80 + "\n")

    print(result.get("answer", "No answer returned."))

    print("\n" + "=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Interactive QA system for PDF books"
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CHECK PDF
    # --------------------------------------------------------

    if not os.path.exists(args.pdf_path):

        print(
            f"❌ Error: File "
            f"'{args.pdf_path}' not found"
        )

        return

    # --------------------------------------------------------
    # CREATE QA SYSTEM
    # --------------------------------------------------------

    try:

        qa_system = create_book_qa_system(
            args.pdf_path
        )

    except Exception as e:

        print("\n❌ Failed to initialize QA system.")

        print(f"\nError: {e}")

        print("\nMake sure:")

        print("1. Your Python packages are installed.")

        print(
            "2. LM Studio is running on "
            "http://localhost:1234"
        )

        print(
            "3. A model is loaded in LM Studio."
        )

        raise

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    chat_history = []

    print(
        "\n📚 Ready to answer questions "
        "about your PDF!"
    )

    print("Type 'quit' to exit.")

    print(
        "Type 'clear' to clear the conversation history."
    )

    # --------------------------------------------------------
    # INTERACTIVE LOOP
    # --------------------------------------------------------

    while True:

        try:

            question = input("\n❓ Ask a question: ").strip()

        except KeyboardInterrupt:

            print("\n\n👋 Goodbye!")

            break

        except EOFError:

            print("\n\n👋 Goodbye!")

            break

        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not question:

            continue

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        if question.lower() in {
            "quit",
            "exit",
            "q",
        }:

            print("\n👋 Goodbye!")

            break

        # ----------------------------------------------------
        # CLEAR HISTORY
        # ----------------------------------------------------

        if question.lower() == "clear":

            chat_history = []

            print(
                "\n🧹 Conversation history cleared."
            )

            continue

        # ----------------------------------------------------
        # ASK QUESTION
        # ----------------------------------------------------

        print(
            "\n🔄 Searching the book "
            "and generating answer..."
        )

        try:

            result = qa_system.invoke(
                {
                    "input": question,
                    "chat_history": chat_history,
                }
            )

            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            print_result(result)

            # ------------------------------------------------
            # SAVE CHAT HISTORY
            # ------------------------------------------------

            answer = result.get(
                "answer",
                "",
            )

            chat_history.append(
                HumanMessage(
                    content=question
                )
            )

            chat_history.append(
                AIMessage(
                    content=answer
                )
            )

        except Exception as e:

            print(
                "\n❌ Error while answering question:"
            )

            print(e)

            print(
                "\nPlease check that LM Studio "
                "is running and a model is loaded."
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
