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

    big_links = []
    URL = "https://ww2.nycourts.gov/ip/adr/index.shtml"
    links = get_links(URL)
    
    big_links = list_check_add(links, big_links)

    
    #only going total depth of 2 pages right now
    i = 0
    while i < 2:  #switch this to 3 when you figure out why the docs aren't processing
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
    
    print("PROCESSING INTO DOCUMENTS")
        
        
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    browser = mechanicalsoup.StatefulBrowser()
    
    loaders=UnstructuredURLLoader(urls=big_links, headers=headers)
    documents=loaders.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": "cpu"})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
      
      
       
main()
