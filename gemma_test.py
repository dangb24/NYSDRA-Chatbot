from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm  

def main():
    model_path = "gemma-2b-it"  # GEMMA模型的路径
    tokenizer = AutoTokenizer.from_pretrained(model_path)  
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_files = ["difficult.txt"]  # 输入文件列表

    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as infile:
            questions = infile.readlines()
        
        output_file = input_file.replace(".txt", "_out.txt")
        with open(output_file, "w", encoding="utf-8") as outfile:
            for question in tqdm(questions, desc=f"Processing {input_file}"):
                question = question.strip()
                if not question:
                    continue  # 跳过空行
                
                # 为每个问题构造输入
                prompt = question
                
                tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                
                outputs = model.generate(**tokens,max_new_tokens=8000)
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                outfile.write(f"Question: {question}\n")
                outfile.write(f"Answer: {output_text}\n\n")

if __name__ == "__main__":
    main()
