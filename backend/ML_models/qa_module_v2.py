# backend/ML_models/qa_module.py

import pandas as pd
import torch

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Import your DocumentParser to use the real PDF processing pipeline
from backend.data_processing.document_parser import DocumentParser

class QAModule:
    """
    A self-contained module for the LangChain-powered Q&A feature.
    It uses only self-hosted, open-source models.
    """
    def __init__(self):
        print("Initializing Q&A Module...")
        # 1. Initialize the self-hosted LLM for generation
        model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Using device_map="auto" to intelligently use GPU if available
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
        hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print(f" - Loaded LLM: {model_id}")

        # 2. Initialize the self-hosted model for embeddings
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # Specify device for embeddings to use GPU if available
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)
        print(f" - Loaded Embedding Model: {embedding_model_name}")
        
        self.vector_store = None
        self.qa_chain = None
        print("Q&A Module initialized successfully.")

    def create_knowledge_base(self, parsed_document: dict, comments_df: pd.DataFrame):
        """
        Creates the searchable knowledge base from the notice and comments.
        """
        print("\nCreating knowledge base from real data...")
        # 1. Prepare documents from the parsed notice PDF
        documents = [Document(page_content=text, metadata={"source": f"Notice Section: {section}"}) 
                     for section, text in parsed_document.items()]
        
        # 2. Prepare documents from the comments DataFrame
        for index, row in comments_df.iterrows():
            content = (f"Comment from {row['UserName']} ({row['OrganizationName'] or 'Individual'}) "
                       f"regarding Section '{row['SectionID']}':\n{row['CommentText']}")
            documents.append(Document(page_content=content, metadata={"source": f"Comment ID: {row['CommentID']}"}))

        # 3. Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        all_splits = text_splitter.split_documents(documents)
        print(f" - Split {len(documents)} source documents into {len(all_splits)} chunks.")
        
        # 4. Create the FAISS Vector Store from the chunks
        print(" - Building FAISS index... (This may take a moment)")
        self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
        print("Knowledge base created and indexed successfully.")

        # 5. Create the final Q&A chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        print("Q&A Chain is ready.")

    def ask_question(self, question: str) -> str:
        """
        Takes a user's question and gets a synthesized answer from the LLM.
        """
        if not self.qa_chain:
            return "Error: Knowledge base has not been created yet."
        
        print(f"\nAsking question: {question}")
        try:
            response = self.qa_chain.invoke(question)
            return response['result']
        except Exception as e:
            return f"An error occurred: {e}"

# --- Main script to demonstrate the full, REAL workflow ---
if __name__ == '__main__':
    # --- THIS SECTION IS NOW UPDATED TO USE YOUR REAL FILES ---

    # 1. Define paths to your real data files
    NOTICE_PDF_PATH = "C:\\Users\\Vansh\\Desktop\\SIH\\Project\\eConsultation-Analyzer\\backend\\data_processing\\Public-notice-bilingual-languge-20250721.pdf"
    COMMENTS_CSV_PATH = "C:\\Users\\Vansh\\Desktop\\SIH\\Project\\dummy.csv"

    # 2. Initialize the main Q&A module (this will load the AI models)
    qa_module = QAModule()
    
    # 3. Use the DocumentParser to process the real PDF
    print("\n--- Parsing Real PDF Document ---")
    try:
        parser = DocumentParser()
        with open(NOTICE_PDF_PATH, "rb") as f:
            real_parsed_doc = parser.parse(f)
        print(f"Successfully parsed {len(real_parsed_doc)} sections from the notice.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Notice PDF not found at '{NOTICE_PDF_PATH}'. Please update the path.")
        exit()

    # 4. Load the full dummy.csv dataset
    print("\n--- Loading Real Comments CSV ---")
    try:
        real_comments_df = pd.read_csv(COMMENTS_CSV_PATH)
        print(f"Successfully loaded {len(real_comments_df)} comments.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Comments CSV not found at '{COMMENTS_CSV_PATH}'. Please update the path.")
        exit()

    # 5. Build the knowledge base using the REAL data
    qa_module.create_knowledge_base(real_parsed_doc, real_comments_df)

    # 6. Ask more relevant and complex questions based on the real data
    answer1 = qa_module.ask_question("What are the main concerns about systemic risk?")
    print(f"Answer 1: {answer1}")

    answer2 = qa_module.ask_question("Summarize the suggestions made by law firms like AZB & Partners.")
    print(f"Answer 2: {answer2}")

    answer3 = qa_module.ask_question("What is the general opinion regarding the ease of doing business?")
    print(f"Answer 3: {answer3}")