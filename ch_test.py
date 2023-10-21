from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_path = "Llama-2-b-chat-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    conversation_history = ""
    max_tokens = 2048

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'end':
            break  # Exit the loop if the user inputs 'end'
        conversation_history += f"You: {user_input}\n"
        system_prompt = "prompt:You are a legal assistant who helps users solve conflict in their daily lives and prevent them from becoming more serious."
        full_prompt = system_prompt + conversation_history + "Assistant: "
        
        # Tokenize the prompt and check the length
        tokens = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        # Generate a response
        outputs = model.generate(**tokens)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Assistant: {output_text}")

        # Append the output to the conversation history
        conversation_history += f"Assistant: {output_text}\n"
        
        # Truncate the conversation history if it exceeds the max tokens
        while True:
            encoded_history = tokenizer.encode(conversation_history, return_tensors="pt")
            if len(encoded_history[0]) > max_tokens:
                # Find the first occurrence of the newline character and remove the text before it
                conversation_history = conversation_history[conversation_history.index('\n') + 1:]
            else:
                break

if __name__ == "__main__":
    main()
