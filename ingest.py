from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader

import torch
import requests
import mechanicalsoup
from bs4 import BeautifulSoup
import time
import xml.etree.ElementTree as ET
import os

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

DB_FAISS_PATH = "vectorstores/db_faiss"
DATA_PATH = "data/"

def listOfCenters(browser, headers):
    delay = 2
    time.sleep(delay)
    page = browser.get('https://www.nysdra.org/centers', headers=headers, timeout=5)
    time.sleep(delay)

    listOfCenters = set()

    # Find all links and extract their href attributes
    links = page.soup.find_all('a')
    for link in links:
        href = link.get('href')
        if href and ".pdf" in href:
            if not os.path.exists(DATA_PATH):
                os.makedirs(DATA_PATH)

            response = requests.get(href)
            filename = href[href.rfind("/") + 1:]
            filepath = os.path.join(DATA_PATH, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)

        elif href and "http" in href and href.lower().find("nysdra") == -1 \
        and href.lower().find("youtube") == -1 and href.lower().find("linkedin") == -1 and href.lower().find("map") == -1:
            listOfCenters.add(href)

    print(len(listOfCenters))
    return list(listOfCenters)

def createVectorDB():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    browser = mechanicalsoup.StatefulBrowser()
    URLs = listOfCenters(browser, headers)

    print(URLs)

    loaders=UnstructuredURLLoader(urls=URLs, headers=headers)
    documents=loaders.load()
    #doc = Document(page_content="text", metadata={"source": "url"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": DEVICE})

    db = FAISS.from_documents(texts, embeddings)

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)

    dbpdf = FAISS.from_documents(texts, embeddings)

    # db.merge_from(dbpdf)

    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    createVectorDB()