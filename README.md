# NYSDRA-Chatbot
A conversational chatbot that uses llama-2-7b-chat.ggmlv3.q8_0.bin model, langchain, and FAISS to read info from passed in URLs and answers questions based on the info it has learned from the URLs.

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