from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def createVectorDB():
    # loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    # documents = loader.load()
    URLs=[
    'https://www.britannica.com/animal/penguin',
    'https://www.livescience.com/animals/birds/penguins',
    'https://www.nationalgeographic.com/animals/birds/facts/penguins-1',
    'https://ocean.si.edu/ocean-life/seabirds/penguins'

    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    loaders=UnstructuredURLLoader(urls=URLs, headers=headers)
    documents=loaders.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)
    #I think you have to do {"device": "cuda"} in order to use GPU
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": "cpu"})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    createVectorDB()