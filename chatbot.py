from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import textwrap
import sys
import os
import torch
import nltk
import subprocess



# pip install python-libmagic
# pip install python-magic-bin
# pip install accelerate
# pip install git+https://github.com/huggingface/transformers.git
# pip install transformers==4.30
# https://github.com/facebookresearch/xformers#installing-xformers


def huggingface_cli_login(token):
    command = f'huggingface-cli login --token {token}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)


# os.environ['OPENAI_API_KEY']='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'

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

URLs=[
    'https://www.britannica.com/animal/penguin',
    'https://www.livescience.com/animals/birds/penguins',
    'https://www.nationalgeographic.com/animals/birds/facts/penguins-1',
    'https://ocean.si.edu/ocean-life/seabirds/penguins'

]

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
    "model.layers":"cpu" ,
    "model.norm":"cpu"
}


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

loaders=UnstructuredURLLoader(urls=URLs, headers=headers)
data=loaders.load()
     

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(data) 
#data is list of langchain.schema.document.Document objects which can be created this way 
#from langchain.docstore.document import Document
#doc = Document(page_content="text", metadata={"source": "url"})

embeddings=HuggingFaceEmbeddings()

query_result = embeddings.embed_query("Hello World") #This is a test "Hello world" is the question being asked

#Convert text_chunks to embeddings and store in FAISS
vectorstore=FAISS.from_documents(text_chunks, embeddings)

# llm=ChatOpenAI()
huggingface_cli_login("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")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                          use_auth_token=True)


model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                             device_map="auto",
                                             torch_dtype=torch.float16,
                                             use_auth_token=True,
                                              load_in_8bit=True,
                                              quantization_config=quantization_config
                                              #load_in_4bit=True
                                             )


pipe = pipeline("text-generation",
                model=model,
                tokenizer= tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens = 512,
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
                )
                
llm=HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0})

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

result=chain({"question": "What is a penguin?"}, return_only_outputs=True)
     
print(result['answer'])