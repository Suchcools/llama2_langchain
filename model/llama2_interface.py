from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

class TextGenerator:
    def __init__(self, model_name, device="cuda:0"):
        self.model = AutoGPTQForCausalLM.from_quantized(model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.device = device
    
    def generate_text(self, prompt):
        input_ids = self.tokenizer([f'<s>Human: {prompt}\n</s><s>Assistant: '], return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.3,
            "repetition_penalty": 1.3,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        
        generate_ids = self.model.generate(**generate_input)
        text = self.tokenizer.decode(generate_ids[0])
        
        return text

# 创建TextGenerator实例
model_name = 'FlagAlpha/Llama2-Chinese-13b-Chat-4bit'
text_generator = TextGenerator(model_name)

# 生成文本
prompt = "它登上了泰山 把这句话拆成主谓宾"
generated_text = text_generator.generate_text(prompt)
print(generated_text)
