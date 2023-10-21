import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, tokenizer, folder_path, block_size, file_chunk_size):
        self.tokenizer = tokenizer
        self.folder_path = folder_path
        self.block_size = block_size
        self.file_chunk_size = file_chunk_size
        self.current_chunk = 0
        self.files = os.listdir(folder_path)
        self.load_next_chunk()

    def load_next_chunk(self):
        self.examples = []
        for i in range(self.current_chunk, min(self.current_chunk + self.file_chunk_size, len(self.files))):
            file_path = os.path.join(self.folder_path, self.files[i])
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.read().splitlines()
                self.examples.extend(lines)
        self.current_chunk += self.file_chunk_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if i == len(self.examples) - 1 and self.current_chunk < len(self.files):
            # Load next chunk of files when the last item of the current chunk is accessed
            self.load_next_chunk()
        return self.tokenizer(self.examples[i], truncation=True, max_length=self.block_size, return_tensors='pt')

def main():
    folder_path = 'your_data_folder'
    model_path = "Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    dataset = CustomDataset(tokenizer, folder_path, block_size=128, file_chunk_size=10)

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        output_dir='./results',
        num_train_epochs=3,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()