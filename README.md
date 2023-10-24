# NYSDRA-Chatbot
A conversational chatbot that uses llama-2-7b-chat.ggmlv3.q8_0.bin model, langchain, and FAISS to use info stored in the vector database to answer questions. The data relates to dispute resolution, mediation, and NYSDRA and the CDRCS in NY.

## How to Run:
Run ingest.py in a linux terminal in order to first create the vector store locally. Then run:
+ **Chainlit**: chainlit run model.py -w
## Some required installations:
+ llama-2-7b-chat.ggmlv3.q8_0.bin
+ pypdf
+ langchain
+ torch
+ accelerate
+ bitsandbytes
+ transformers
+ sentence_transformers
+ faiss_cpu
+ chainlit
