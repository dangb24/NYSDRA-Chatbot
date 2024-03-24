from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_path = "gemma-2b-it"  # 更新为GEMMA模型路径
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    conversation_history = ""
    max_tokens = 8000

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'end':
            break
        
        # 构造对话历史记录，GEMMA模型适用的格式
        conversation_history += f"<start_of_turn>user\n{user_input}<end_of_turn>"
        
        # 创建完整的提示
        full_prompt = conversation_history
        
        # 生成响应
        tokens = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        encoded_history = tokenizer.encode(conversation_history, return_tensors="pt")
        if len(encoded_history[0]) > 8000:
 
            cut_index = encoded_history[0].tolist().index(tokenizer.eos_token_id) + 1
            conversation_history = tokenizer.decode(encoded_history[0][cut_index:])
        outputs = model.generate(**tokens, max_new_tokens=8000, no_repeat_ngram_size=2)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Assistant: {output_text}")

        # 更新对话历史以包含助手的回答
        conversation_history += f"<start_of_turn>model\n{output_text}<end_of_turn>"

if __name__ == "__main__":
    main()
