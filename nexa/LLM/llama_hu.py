from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from nexa import LLM_DIR
from nexa.LLM import LargeLanguageModelAbstract


class SambaLingo(LargeLanguageModelAbstract):

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sambanovasystems/SambaLingo-Hungarian-Chat",
            use_fast=False
        )
        self.device = torch.device("mps")  # Metal Performance Shaders (Apple GPU)
        self.model = AutoModelForCausalLM.from_pretrained(
            "sambanovasystems/SambaLingo-Hungarian-Chat",
            torch_dtype=torch.bfloat16
        ).to(self.device)

    def generate(self, text: str) -> str:
        messages = [{"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.0,
                do_sample=True
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
    
        return response

    def pipeline(self, text: str, file_path: Path):
        text = self.generate(text)
        with open(file_path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    text = "Miért esik jól kávét inni reggel?"
    llm = SambaLingo()
    llm.pipeline(text, LLM_DIR / "test_hu.txt")