# NYSDRA-Chatbot
A conversational chatbot that uses llama-2-7b-chat.ggmlv3.q8_0.bin model, langchain, and FAISS to use info stored in the vector database to answer questions. The data relates to dispute resolution, mediation, and NYSDRA and the CDRCS in NY.

## How to Run:
Run ingest.py first in order to create the vector store locally. Then run:
+ **Chainlit**: chainlit run model.py -w
+ **Flask**: python3 model.py
## Some required installations:
+ llama-2-7b-chat.ggmlv3.q8_0.bin (available on huggingface)
+ pypdf
+ langchain
+ torch
+ faiss_cpu
+ chainlit
+ flask

## Things to work on:
+ Utilize chat history within Conversational Retreival Chain
+ Refactor ingest.py
+ Enable chatbot to give appropriate responses to conversational queries like "Hello"
+ Utilize user location for suggesting resources that are closer to the user
