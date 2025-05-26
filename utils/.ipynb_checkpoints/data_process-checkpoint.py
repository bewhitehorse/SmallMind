import torch
from torch.utils.data import Dataset
import tiktoken
import json
from config import SmallMindConfig

torch.manual_seed(1024)


class MyDataset(Dataset):
    def __init__(self, path, config):

        self.enc = tiktoken.get_encoding('gpt2')
        self.block_size = config.block_size

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]
    

        self.max_lines = 1000
        raw_data = []
        with open(path,'r') as f:
            for i, line in enumerate(f):
                if i >=  self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue

        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])

        self.encoded_data = []
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        return self.enc.encode(text)
    
    def decode(self, ids):
        return self.enc.decode(ids)

def ReturnQuestionTensor(question, config):
    enc = tiktoken.get_encoding('gpt2')
    encoded_text = enc.encode(question)
    eos_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    full_encoded = encoded_text + [eos_token]  
    assert len(full_encoded) <= config.block_size, "Question is too long"

    if len(full_encoded) < config.block_size + 1:
        full_encoded = full_encoded + [eos_token] * (config.block_size + 1 - len(full_encoded))

    return torch.tensor(full_encoded, dtype=torch.long)