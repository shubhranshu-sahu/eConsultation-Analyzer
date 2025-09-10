# backend/ML_models/qa_module.py

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

class QAModule:
    """
    A self-contained module for the LangChain-powered Q&A feature.
    It uses only self-hosted, open-source models.
    """
    def __init__(self):
        print("Initializing Q&A Module...")
        # 1. Initialize the self-hosted LLM for generation
        # Using Flan-T5-Large as it's powerful and permissively licensed
        model_id = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
        print(f" - Loaded LLM: {model_id}")

        # 2. Initialize the self-hosted model for embeddings
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        print(f" - Loaded Embedding Model: {embedding_model_name}")
        
        self.vector_store = None
        self.qa_chain = None
        print("Q&A Module initialized successfully.")

    def create_knowledge_base(self, parsed_document: dict, comments_df: pd.DataFrame):
        """
        Creates the searchable knowledge base from the notice and comments.
        """
        print("\nCreating knowledge base...")
        # 1. Prepare documents from the parsed notice PDF
        documents = [Document(page_content=text, metadata={"source": f"Notice Section: {section}"}) 
                     for section, text in parsed_document.items()]
        
        # 2. Prepare documents from the comments DataFrame
        for index, row in comments_df.iterrows():
            # Combine comment text with rich metadata for better context
            content = (f"Comment from {row['UserName']} ({row['OrganizationName'] or 'Individual'}) "
                       f"regarding Section '{row['SectionID']}':\n{row['CommentText']}")
            documents.append(Document(page_content=content, metadata={"source": f"Comment ID: {row['CommentID']}"}))

        # 3. Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        all_splits = text_splitter.split_documents(documents)
        
        # 4. Create the FAISS Vector Store from the chunks
        self.vector_store = FAISS.from_documents(all_splits, self.embeddings)
        print("Knowledge base created and indexed successfully.")

        # 5. Create the final Q&A chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # "Stuff" chain type passes all retrieved docs into the prompt
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
        response = self.qa_chain.invoke(question)
        return response['result']

# --- Main script to demonstrate the full workflow ---
if __name__ == '__main__':
    # 1. Initialize the module (this will download models on first run)
    qa_module = QAModule()

    # 2. Create mock data (in a real app, this comes from your other modules)
    mock_parsed_doc = {
        "Rule 11(2)(b)": "The amendment proposes to insert a clause regarding Finance Companies registered with the IFSCA...",
        "Preamble": "This notice is to invite suggestions on the draft amendment aimed at providing ease of doing business for Finance Companies."
    }
    
    mock_comments_data = {
        'CommentID': [1003, 1006],
        'UserName': ['Ravi Kumar', 'Sunita Rao'],
        'OrganizationName': [None, 'AZB & Partners'],
        'SectionID': ['Rule 11(2)', 'Rule 11(2)(b)'],
        'CommentText': [
            "This blanket exemption could create a systemic risk without proper oversight.",
            "The language in Rule 11(2)(b) is currently ambiguous. We suggest adding a clause to protect small businesses."
        ]
    }
    mock_comments_df = pd.DataFrame(mock_comments_data)

    # 3. Build the knowledge base
    qa_module.create_knowledge_base(mock_parsed_doc, mock_comments_df)

    # 4. Ask questions
    answer1 = qa_module.ask_question("What is the main concern about the exemption?")
    print(f"Answer 1: {answer1}")

    answer2 = qa_module.ask_question("What was the suggestion for Rule 11(2)(b)?")
    print(f"Answer 2: {answer2}")