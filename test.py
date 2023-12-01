from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import torch.quantization
from tqdm import tqdm  

def main():
    model_path = "Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_files = ["easy.txt"]  # List of input files

    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as infile:
            questions = infile.readlines()
        
        output_file = input_file.replace(".txt", "_out.txt")
        with open(output_file, "w", encoding="utf-8") as outfile:
            for question in tqdm(questions, desc=f"Processing {input_file}"): 
                question = question.strip()
                if not question:
                    continue  # Skip empty lines
                
                system_prompt = "<s>[INST] <<SYS>>\nYou are a legal assistant who helps users solve conflict in their daily lives and prevent them from becoming more serious.\n<</SYS>>\n\n"
                user_input = f"{question} [/INST]"
                full_prompt = system_prompt + user_input
                
                tokens = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                
                outputs = model.generate(**tokens)
                start_idx = len(tokens['input_ids'][0])  # Get the number of input tokens
                output_text = tokenizer.decode(outputs[0][start_idx:], skip_special_tokens=True)  # Start decoding from after the input tokens

                outfile.write(f"Question: {question}\n")
                outfile.write(f"Answer: {output_text}\n\n")

if __name__ == "__main__":
    main()
