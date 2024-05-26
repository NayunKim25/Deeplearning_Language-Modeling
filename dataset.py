import torch
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, input_file='C:/Users/nayun/Downloads/Laguage/shakespeare_train.txt', seq_len=30):
        self.seq_len = seq_len
        self.vocab = self.build_vocab(input_file)
        self.data = self.load_data(input_file)
        self.trn_idx, self.val_idx = self.split_data()
    
    def build_vocab(self, input_file):
        # input 파일에서 고유한 문자를 추출하여 vocabulary를 만듦
        vocab = set()
        with open(input_file, 'r', encoding='utf-8') as f:
            data = f.read()
            vocab.update(data)
        vocab = sorted(vocab)
        vocab_dict = {char: idx for idx, char in enumerate(vocab)}
        return vocab_dict

    def load_data(self, input_file):
        # input 파일에서 데이터를 읽어와서 문자를 인덱스로 변환하여 저장
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data += [self.vocab[char] for char in line]
        return data

    def split_data(self, val_ratio=0.2):
        # train, validation 데이터 셋 구분
        val_size = int(len(self.data) * val_ratio)
        trn_idx = torch.arange(len(self.data) - val_size)
        val_idx = torch.arange(len(self.data) - val_size, len(self.data))
        return trn_idx, val_idx

    def __len__(self):
        # 데이터셋 길이 반환(시퀀스 길이 뺀 길이)
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 인덱스에서 시퀀스와 타겟 시퀀스 반환
        input_seq = self.data[idx:idx+self.seq_len]
        target_seq = self.data[idx+1:idx+self.seq_len+1]
    
        input_len = len(input_seq)
        input_tensor = torch.zeros(self.seq_len).long()
        input_tensor[:input_len] = torch.tensor(input_seq)
    
        target_len = len(target_seq)
        target_tensor = torch.zeros(self.seq_len).long()
        target_tensor[:target_len] = torch.tensor(target_seq)
    
        return input_tensor, target_tensor

if __name__ == '__main__':
    dataset = ShakespeareDataset()
    print(f"vocab size: {len(dataset.vocab)}")
    print(f"data length: {len(dataset.data)}")
    print(f"train set size: {len(dataset.trn_idx)}")
    print(f"validation set size: {len(dataset.val_idx)}")

    input_seq, target_seq = dataset[0]
    print(f"input_seq: {input_seq}")
    print(f"target_seq: {target_seq}")