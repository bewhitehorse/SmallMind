from config import SmallMindConfig
from model.model import SmallMind
from utils.data_process import ReturnQuestionTensor
import torch
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

config = SmallMindConfig()
model = SmallMind(config)
model.to(device)

checkpoint_path = 'checkpoints/model_epoch_2.pt'  
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

def generate_text(model, input_text, max_length=50):
    model.eval()
    with torch.no_grad():
        input_ids = ReturnQuestionTensor(input_text, config).to(device)
        output_ids = model.generate(input_ids)
        generated_text = enc.decode(output_ids.squeeze().tolist())
        return generated_text

input_text = "你好啊"
generated_text = generate_text(model, input_text)
print(f"Input: {input_text}\nGenerated: {generated_text}")