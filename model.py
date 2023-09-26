from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA #This is just a retrieval chain, for chat history use conversational retrieval chain

# LLMChain: This chain uses a Language Model for generating responses to queries or prompts. 
# It can be used for various tasks such as chatbots, summarization, and more

# StuffDocumentsChain: This is the default QA chain used in the RetrievalQAChain. It processes the retrieved documents 
# and generates answers to questions based on the content of the documents

#RetrievalQAChain: This chain combines a Retriever and a QA chain. 
#It is used to retrieve documents from a Retriever and then use a QA chain to answer a question based on the retrieved documents

# the RetrievalQAChain is used with a VectorStore as the Retriever and the default StuffDocumentsChain as the QA chain.

#From what I understand:
# RetrievalQAChain: retrieves the vector based on the prompt
# StuffDocumentsChain: Converts the vector into the answer
# LLMChain: Uses the answer to generate a response to the prompt


DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def setCustomPrompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def loadLLM():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens=512, temperature=0.5
    )
    return llm

def retrievalQAChain(llm, prompt, db):
    #Look into the different available chain_type
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 2}), return_source_documents = True, 
        chain_type_kwargs={"prompt": prompt}
    )
    #search_kwargs={"k": 2} means 2 searches
    #return_source_documents = True means don't use base knowledge use only knowledge we provided
    return qa_chain

def qaBot():
    #I think you have to do {"device": "cuda"} in order to use GPU
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": "cpu"})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    llm = loadLLM()
    qa_prompt = setCustomPrompt()
    qa = retrievalQAChain(llm, qa_prompt, db)

    return qa

def finalResult(query):
    qa_result = qaBot()
    response = qa_result({"query": query})
    print()
    return response


if __name__ == "__main__":
    while True:
        prompt = input("Please enter your question (or 'q' to quit): ")
        if prompt.lower() == "q":
            break
        print()
        print(finalResult(prompt), end="\n\n")


