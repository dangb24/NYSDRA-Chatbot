from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from tqdm import tqdm  

def main():
    model_path = "Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_files = ["easy.txt", "medium.txt", "difficult.txt"]  # List of input files

    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as infile:
            questions = infile.readlines()
        
        output_file = input_file.replace(".txt", "_out.txt")
        with open(output_file, "w", encoding="utf-8") as outfile:
            for question in tqdm(questions, desc=f"Processing {input_file}"): 
                question = question.strip()
                if not question:
                    continue  # Skip empty lines
                
                system_prompt = "You are a legal assistant who helps users solve conflict in their daily lives and prevent them from becoming more serious."
                full_prompt = system_prompt + question
                
                tokens = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                
                outputs = model.generate(**tokens)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                outfile.write(f"Question: {question}\n")
                outfile.write(f"Answer: {output_text}\n\n")

if __name__ == "__main__":
    main()
