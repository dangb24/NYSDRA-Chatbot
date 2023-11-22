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

def getAllLinksInPage(base_url, url, setOfInsideLinks, setOfWrongLinks, browser, headers, level):
    max_level = 1
    delay = 2
    time.sleep(delay)
    try:
        page = browser.get(url, headers=headers, timeout=5)
        if page == None or page.soup == None:
            setOfWrongLinks.add(url)
            return
        if page.status_code == 404:
            setOfWrongLinks.add(url)
            print(f"404 Not Found: {url}")
            setOfInsideLinks.remove(url)
            return
    except Exception as e:
        print(url)
        print(f"{e}")
        setOfWrongLinks.add(url)
        setOfInsideLinks.remove(url)
        return
    time.sleep(delay)

    # Find all links and extract their href attributes
    links = page.soup.find_all('a')
    links += page.soup.find_all('link')
    for link in links:
        href = link.get('href')

        if href and href[-1] == "/":
            href = href[0:len(href)-1]

        if href and "http" in href:
            continue
        elif href and (base_url + href).rfind("html") == (base_url + href).find("html") and \
        href.rfind("pdf") == -1 and href.rfind("png") == -1 and href.rfind("json") == -1 and href.rfind(":") == -1 and \
        href.rfind(".ico") == -1 and href.rfind(".svg") == -1 and href.rfind(".si") == -1 and href.rfind("?") == -1 and \
        href.rfind("%20") == -1 and href.rfind("#") == -1 and (base_url + href).rfind(".com") == (base_url + href).find(".com"):

            link = ""

            if href[0] != "/" and base_url[-1] != "/":
                link = base_url + "/" + href
            elif href[0] == "/" and base_url[-1] == "/":
                link = base_url + href[1:]
            else:
                link = base_url + href

            if link in setOfWrongLinks or link in setOfInsideLinks:
                continue

            setOfInsideLinks.add(link)

            print("URL: ", link)
            if level < max_level:
                getAllLinksInPage(base_url, link, setOfInsideLinks, setOfWrongLinks, browser, headers, level + 1)

def listOfCenters(browser, headers):
    delay = 2
    time.sleep(delay)
    page = browser.get('https://www.nysdra.org/centers', headers=headers, timeout=5)
    time.sleep(delay)

    listOfCenters = set()

    documentList = []

    # Find all links and extract their href attributes
    links = page.soup.find_all('a')
    iterations = 0
    for link in links:
        href = link.get('href')
        if href and ".pdf" in href:
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
            setOfInsideLinks = set()
            setOfWrongLinks = set()
            setOfInsideLinks.add(href)
            getAllLinksInPage(href, href, setOfInsideLinks, setOfWrongLinks, browser, headers, 0)
            listOfCenters = listOfCenters.union(setOfInsideLinks)
            iterations += 1
        
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

    loaders2 = CSVLoader(file_path= "data/Conversation.csv", encoding="utf-8", csv_args={
                'delimiter': ','})
    documents2 = loaders2.load()
    documents += documents2

    loaders2 = TextLoader(file_path= "data/QA.txt", encoding="utf-8")
    documents2 = loaders2.load()
    documents += documents2

    for document in documents:
        document.page_content = document.page_content.replace("\n", "")

    #doc = Document(page_content="text", metadata={"source": "url"})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
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