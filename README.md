# NYSDRA-Chatbot
The model size is too large to upload to the gihub.

The model can be found on the Huggingface website. Please first apply for usage rights on Meta AI and then download the model using a token.

Replace the model loading section with the following two lines of code to download the model.

tokenizer = AutoTokenizer.from_pretrained("meta-llama/"+ model_path, cache_dir=model_path, use_auth_token="your_token")
model = AutoModelForCausalLM.from_pretrained("meta-llama/"+model_path, cache_dir=model_path,use_auth_token="your_token")
