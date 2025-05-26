import torch
from torch.utils.data import Dataset
import tiktoken
import json

torch.manual_seed(1024)


class MyDataset(Dataset):
    def __init__(self, path, config):

        self.enc = tiktoken.get_encoding('gpt2')
        self.block_size = config.block_size
        self.max_lines = config.max_lines
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]
    
        # 读取和编码数据
        raw_data = self._load_data(path)
        full_encoded = self._encode_data(raw_data)
        self.encoded_data = self._chunk_data(full_encoded)

    def _load_data(self, path):
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
        return raw_data
    
    def _encode_data(self, raw_data):
        # 编码数据
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        return full_encoded
    
    def _chunk_data(self, full_encoded):
        # 将编码后的数据分块
        encoded_data = []
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1]
            if len(chunk) < self.block_size + 1:
                chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
            encoded_data.append(chunk)
        return encoded_data
    
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

    assert len(encoded_text) <= config.block_size, "Question is too long"

    return torch.tensor(encoded_text, dtype=torch.long)