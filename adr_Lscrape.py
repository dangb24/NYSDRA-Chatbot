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
import os

import json

import urllib.request
from urllib.request import urlopen
import html2text
import requests


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
    while i < 3:  #generally keep this at 3 for pdfs
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
            
            try:
                pdf_io_bytes = io.BytesIO(response.content)
                text_list = []
                pdf = PyPDF2.PdfReader(pdf_io_bytes)

                num_pages = len(pdf.pages)

                for page in range(num_pages):
                    page_text = pdf.pages[page].extract_text()
                    text_list.append(page_text)
                text = "\n".join(text_list)
                
                pdfList.append(text)
                print("added pdf")
            
            except:
                print("bad pdf")
            
            
        else:
         
            urlList.add(link)
            

    print("Completed url and pdf splitting")
    print("num pdfs: " + str(len(pdfList)))
    print("num urls: " + str(len(urlList))) 
    

    
    #okay so make the txt files for the pdfs first
    file_index = 1
    txt_list = []
    for pdf in pdfList:
        filename = "file" + str(file_index) + ".txt"
        print(filename)
        
        file = open("txt_files/"+filename, "w")
        
        file.write(pdf)

        file.close()
        
        if os.path.getsize("txt_files/"+filename) == 0:
            os.remove("txt_files/"+filename)
            
        else:   
            txt_list.append(filename)
            
            
            file_index += 1
    
    u_in = input("do you want to read the urls too?  Please enter y/n.  (if you're running this on 3 then that's big and takes a lot of time)")
    
    if u_in == 'y':
         
        headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        browser = mechanicalsoup.StatefulBrowser()     
          
        halvsies = (len(urlList)/2) 
        f_half= []
        i = 0
        while i < halvsies:
            f_half.append(urlList.pop())
            i+=1
             
        print("num in first half: " + str(len(f_half)))
        print("second: " + str(len(urlList)))
        
        
        cont = 1
        count = 1
        while cont == 1 and count <= 2:
            loaders=UnstructuredURLLoader(urls=f_half, headers=headers)
            documents=loaders.load()
            
            for doc in documents:
                filename = "file" + str(file_index) + ".txt"
                print(filename)
                
                file = open("web_files/"+filename, "w")

                file.write(doc.page_content)
                
                file.close()

                txt_list.append(filename) 
                
                
                file_index += 1
            
            count +=1 
            cont = int(input("please input 1 if you want to process the next half of the url pages and 0 if you don't"))  
            print(cont)
            if cont == 1:
                f_half = urlList  
                  
main()