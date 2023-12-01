import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch

class CustomDataset(Dataset):
    def __init__(self, tokenizer, folder_names, root_folder_path, block_size, file_chunk_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.file_chunk_size = file_chunk_size
        self.current_chunk = 0
        self.files = []
        for folder_name in folder_names:
            folder_path = os.path.join(root_folder_path, folder_name)
            self.files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')])
        self.load_next_chunk()

    def load_next_chunk(self):
        self.examples = []
        current_text = ""
        file_index = self.current_chunk
        while file_index < len(self.files) and len(self.examples) < self.file_chunk_size:
            with open(self.files[file_index], 'r', encoding='utf-8') as file:
                for line in file:
                    current_text += line.strip() + " "
                    if len(self.tokenizer.encode(current_text)) >= self.block_size:
                        encoded_text = self.tokenizer.encode(current_text)[:self.block_size]
                        self.examples.append(self.tokenizer.decode(encoded_text))
                        current_text = self.tokenizer.decode(self.tokenizer.encode(current_text)[self.block_size:])
            file_index += 1

        if current_text:
            padding_needed = self.block_size - len(self.tokenizer.encode(current_text))
            current_text += self.tokenizer.decode([0] * padding_needed)
            self.examples.append(current_text)
        
        self.current_chunk = file_index

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if i == len(self.examples) - 1 and self.current_chunk < len(self.files):
            self.load_next_chunk()
        
        tokenized_text = self.tokenizer(self.examples[i], truncation=True, max_length=self.block_size, return_tensors='pt')

        
        if tokenized_text['input_ids'].shape[1] < self.block_size:
            padding_needed = self.block_size - tokenized_text['input_ids'].shape[1]
            padded_input_ids = torch.cat([tokenized_text['input_ids'], torch.full((1, padding_needed), 0)], dim=1)
            padded_attention_mask = torch.cat([tokenized_text['attention_mask'], torch.full((1, padding_needed), 0)], dim=1)
        else:
            padded_input_ids = tokenized_text['input_ids']
            padded_attention_mask = tokenized_text['attention_mask']

        return {'input_ids': padded_input_ids, 'attention_mask': padded_attention_mask}

def custom_train(model, data_loader, device, epochs, learning_rate):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.train()
    
    for epoch in range(epochs):
        for step, batch in enumerate(data_loader):
 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            input_ids = batch['input_ids'].squeeze(1).to(device) 
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(step)
            if step % 10 == 0:  
                print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

def main():
    folder_names = ['txt_files', 'web_files']
    root_folder_path = ''  # Provide the path to your data
    model_path = "Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    dataset = CustomDataset(tokenizer, folder_names, root_folder_path, block_size=4096, file_chunk_size=10)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(len(dataset))
    # Start custom training
    custom_train(model, data_loader, device, epochs=3, learning_rate=5e-5)

if __name__ == "__main__":
    main()