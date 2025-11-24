import pinecone
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone with your API key
pc = Pinecone(api_key="pcsk_VNtHX_61QnFVhjH9PwaxUE8cdAcBoUCrbYEbpyxh2cPPrtQAfvuuNB2crbvFZwipALYU6")

# If you already created an index in your Pinecone dashboard, you can skip this step.
# Otherwise, create a new one:
pc.create_index(
    name="test-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Connect to the index
index = pc.Index("test-index")

print("✅ Pinecone is installed, configured, and your index is ready!")
