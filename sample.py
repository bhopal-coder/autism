from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA

# 1. Initialize Pinecone
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")

index_name = "pdf-chatbot-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# 2. Load your PDF
loader = PyMuPDFLoader("your_file.pdf")
docs = loader.load()

# 3. Convert text to embeddings
embeddings = OpenAIEmbeddings(api_key="YOUR_OPENAI_API_KEY")

# 4. Store in Pinecone
vector_store = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

# 5. Build the chatbot
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key="YOUR_OPENAI_API_KEY", model="gpt-3.5-turbo"),
    retriever=vector_store.as_retriever()
)

# 6. Ask your chatbot
query = "Summarize the main points from this PDF."
print(qa.run(query))
