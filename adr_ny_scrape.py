import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredURLLoader
import mechanicalsoup
from bs4 import BeautifulSoup
import time
import xml.etree.ElementTree as ET
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import io

import PyPDF2
import torch
import os
import csv

DB_FAISS_PATH = "vectorstores/db_faiss"

#check for duplicate links
def list_check_add(a_list, big_list):
    for element in a_list:
        if element not in big_list:
            big_list.append(element)
    
    return big_list



def get_links(URL):
    # print("HELLO")
    try:
        page = requests.get(URL, verify = False)
    
    except:
        print("bad link")
        return


    soup = BeautifulSoup(page.content, "html.parser")

    results = soup.find(id="block-nycourts-content")
    
    pre_links = []
    if(not bool(results)):
        return pre_links

    pre_links = results.find_all("a")
    linksHopefully = []
    
    for link in pre_links:
        if str(link).find("href") != -1:
            if str(link).find("http") != -1:
                linksHopefully.append(link)
 

    #process the links a little bit
    linkList = []
    if not bool(linksHopefully):
        print("NO LINKS")
        return linkList
    
    else:
        
        for link in linksHopefully:
            linkList.append(link["href"])

        return linkList
    




def main():
    
    
    #first scraping all of the links from however many layers of internet you want to go through
    big_links = []
    URL = "https://ww2.nycourts.gov/ip/adr/index.shtml"
    links = get_links(URL)
    
    big_links = list_check_add(links, big_links)

    
    #only going total depth of 2 pages right now
    i = 0
    while i < 3:  #switch this to 3 when you figure out why the docs aren't processing
        subs = []
        if not bool(list):
            break
        
        for link in links:
            sub_links = get_links(link)
            if bool(sub_links):
                subs += sub_links
   
        if bool(subs):
             big_links = list_check_add(subs, big_links)
             links = subs
        i += 1  
    

    print()
    print("Final list:")
    for element in big_links:
        print(element)  
        
        
        
    print()
    
    
    
   #next split them into url links and pdfs so that the pdfs can be read properly 
    print("PROCESSING INTO DOCUMENTS")
           
   
    
    pdfList= []
    urlList = set()
    
    #also some pdfs are screwy bc of not finding EOF so here is some error handling
    EOF_MARKER = b'%%EOF'
    
    for link in big_links:
        if ".pdf" in str(link):
            print("found pdf: " + link)
            
            response = requests.get(link)
            # with open("temp.pdf", "wb") as f:
            #     f.write(response.content)

            # pdf_file = open("temp.pdf", "rb")
            # print(pdf_file.read())
            # if EOF_MARKER in pdf_file:
            #     #  pdf_file = pdf_file + EOF_MARKER
            #     reader = PyPDF2.PdfReader(pdf_file)
            #     text = ""
                
            #     # pdf_file = open("temp.pdf", "rb")
            #     # reader = PyPDF2.PdfReader(pdf_file)
            #     # text = ""
            #     for num in range(len(reader.pages)):
            #         page = reader.pages[num]
            #         text += page.extract_text()

            #     pdf_file.close()
            #     os.remove("temp.pdf")
            
            
            try:
                pdf_io_bytes = io.BytesIO(response.content)
                text_list = []
                pdf = PyPDF2.PdfReader(pdf_io_bytes)

                num_pages = len(pdf.pages)

                for page in range(num_pages):
                    page_text = pdf.pages[page].extract_text()
                    text_list.append(page_text)
                text = "\n".join(text_list)
                
                pdfList.append(Document(page_content=text.replace("\n", "").replace("\x00", "f"), metadata={"source": link}))
                print("added pdf")
            
            except:
                print("bad pdf")
            
            
           
            # else:
            #     print("no EOF")
            #     pdf_file.close()

           
            
        else:
            urlList.add(link)
            
    print("Completed url and pdf splitting")
    print("num pdfs: " + str(len(pdfList)))
    print("num urls: " + str(len(urlList))) 
       
    #and here's the fun part where things usually start to break
           
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    browser = mechanicalsoup.StatefulBrowser()     
        
    
    loaders=UnstructuredURLLoader(urls=urlList, headers=headers)
    documents=loaders.load()
    documents+=pdfList
    
    for document in documents:
        document.page_content = document.page_content.replace("\n", "")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents) 
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": "cpu"})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
      
      
       
main()
