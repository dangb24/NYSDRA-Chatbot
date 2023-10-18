from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader, UnstructuredURLLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

import PyPDF2
import torch
import requests
import mechanicalsoup
from bs4 import BeautifulSoup
import time
import xml.etree.ElementTree as ET
import os
import csv


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

DB_FAISS_PATH = "vectorstores/db_faiss"

def listOfCenters(browser, headers):
    delay = 2
    time.sleep(delay)
    page = browser.get('https://www.nysdra.org/centers', headers=headers, timeout=5)
    time.sleep(delay)

    listOfCenters = set()

    documentList = []

    # Find all links and extract their href attributes
    links = page.soup.find_all('a')
    for link in links:
        href = link.get('href')
        if href and ".pdf" in href:
            print(href)

            response = requests.get(href)
            with open("temp.pdf", "wb") as f:
                f.write(response.content)

            pdf_file = open("temp.pdf", "rb")
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for num in range(len(reader.pages)):
                page = reader.pages[num]
                text += page.extract_text()

            pdf_file.close()
            os.remove("temp.pdf")

            documentList.append(Document(page_content=text.replace("\n", "").replace("\x00", "f"), metadata={"source": href}))

        elif href and "http" in href and href.lower().find("nysdra") == -1 \
        and href.lower().find("youtube") == -1 and href.lower().find("linkedin") == -1 and href.lower().find("map") == -1:
            listOfCenters.add(href)

    print(len(listOfCenters))
    return (list(listOfCenters), documentList)

def createVectorDB():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    browser = mechanicalsoup.StatefulBrowser()
    infoTuple = listOfCenters(browser, headers)
    URLs = infoTuple[0]
    pdfDocumentList = infoTuple[1]

    print(URLs)

    loaders=UnstructuredURLLoader(urls=URLs, headers=headers)
    documents=loaders.load()
    documents += pdfDocumentList

    loaders2 = CSVLoader(file_path= "./Conversation2.csv", encoding="utf-8", csv_args={
                'delimiter': ','})
    documents2 = loaders2.load()
    documents += documents2

    # loader2 = TextLoader(file_path="./human_chat.txt")
    # document2 = loader2.load()
    # documents += document2

    for document in documents:
        document.page_content = document.page_content.replace("\n", "")

    #doc = Document(page_content="text", metadata={"source": "url"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": DEVICE})

    db = FAISS.from_documents(texts, embeddings)

    db.save_local(DB_FAISS_PATH)

    

    # with open('Conversation.csv', 'r', newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     lines = list(reader)
    # processed_lines = []
    # for line in lines:
    #     if line:
    #         newList = list()
    #         newList.append(line[1])
    #         newList.append(line[2])
    #         processed_lines.append(newList)

    # print(processed_lines)
    # with open('Conversation.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerows(processed_lines)

if __name__ == "__main__":
    createVectorDB()