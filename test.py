from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers

llm = CTransformers(model='model/llama-2-7b.ggmlv3.q4_K_M.bin', model_type='llama')

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

vectorstore = Chroma.from_documents(documents=all_splits, embedding=emb)

question = "What are the approaches to Task Decomposition?"
question = "What are challenges in building LLM-centred agents?"


docs = vectorstore.similarity_search(question)
len(docs)

qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
qa_chain({"query": question})


# index = VectorstoreIndexCreator().from_loaders([loader])