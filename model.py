from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA #This is just a retrieval chain, for chat history use conversational retrieval chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory  import ConversationTokenBufferMemory

from typing import Dict, Any
import chainlit as cl
import torch

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

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.
Be empathetic, sympathetic, and kind in your responses.


Context: {chat_history} 
{context}

Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""

custom_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question. 
Preserve the original question in the answer sentiment during rephrasing.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT_CUSTOM = PromptTemplate(template=custom_template, input_variables=["question", "chat_history"])


def setCustomPrompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question", "chat_history"])
    return prompt

def loadLLM():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens=512, temperature=0)
    return llm

def retrievalQAChain(llm, prompt, db):
    # memory = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, input_key="question", max_token_limit=512)
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 1}), return_source_documents = True, 
    #     chain_type_kwargs={
    #         "verbose": False,
    #         "prompt": prompt,
    #         "memory": memory,
    #     }, verbose=True
    # )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever= db.as_retriever(search_kwargs={"k": 1}), 
        combine_docs_chain_kwargs={"prompt": prompt}, 
        return_source_documents = True, 
        verbose=True,
        # condense_question_prompt=CONDENSE_QUESTION_PROMPT_CUSTOM,
        rephrase_question = False
        )
    #search_kwargs={"k": 2} means 2 searches
    #return_source_documents = True means don't use base knowledge use only knowledge we provided
    return qa_chain

def qaBot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs={"device": DEVICE})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    llm = loadLLM()
    qa_prompt = setCustomPrompt()
    qa = retrievalQAChain(llm, qa_prompt, db)

    return qa

def finalResult(query):
    qa_result = qaBot()
    chat_history = []
    # Will be query if using RetrievalQA, question for ConversationalQA
    response = qa_result({'chat_history': chat_history, 'question': query})
    print()
    return response


@cl.on_chat_start
async def start():
    bot = qaBot()
    await cl.Message(content="Hello, Welcome to the ChatBot. What is your question?").send()
    cl.user_session.set("chatbot", bot)

@cl.on_message
async def main(message):
    bot = cl.user_session.get("chatbot")
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    chat_history = []
    result = await bot.acall({'chat_history': chat_history, 'question': message}, callbacks=[cb])
    # Will be result["result"] if using RetrievalQA result["answer"] for ConversationalQA
    answer = result["answer"]
    sources = result["source_documents"]

    if sources:
        answer += f"\n\nSources:" + str(sources)
    else:
        answer += f"\n\nNo Sources Found"
    

    await cl.Message(content=answer).send()


#This is how to run with chainlit: chainlit run model.py -w

if __name__ == "__main__":
    while True:
        prompt = input("Please enter your question (or 'q' to quit): ")
        if prompt.lower() == "q":
            break
        print()
        print(finalResult(prompt), end="\n\n")