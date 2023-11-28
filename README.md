# NYSDRA-Chatbot


## Components


### adr_Lscrape
Contains the program for scrapping the adr website and processing it into txt files.  The files are stored in web_files and txt_files.

### adr_ny_scrape
This scrapes the adr website and processes the data into Chainlit documents.  The documents are stored in vectorstores.

### csv_process
Processes the data from policy_test_data and policy_train_data.  The processed data is stored in processed_csv.



## collected data
* vectorstores 
  - Chainlit documents of data scrpaed from the ADR NYS website
 * processed_csv
   - processed data from policy_test and policy_train in seperate files.  Useful for trianing in semi-legal conversational style.
* txt_files
  - data from pdfs scrapes from the ADR NYS website
* web_files
  - data form the websites themselves scraped from the ADR NYS website

