import torch
import random
from model import CharLSTM, CharRNN
from dataset import ShakespeareDataset

def generate(model, dataset, seed_characters, temperature=1.0, max_length=100):
    """
    Generate characters

    Args:
        model: trained model
        dataset: dataset object containing vocabulary mappings
        seed_characters: seed characters
        temperature: temperature for sampling
        max_length: maximum length of generated sequence

    Returns:
        samples: generated characters
    """
    model.eval()

    # 시드 문자를 텐서로 변환
    seed_tensor = torch.tensor([dataset.vocab[char] for char in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    generated = seed_characters
    
    # 문자 생성
    with torch.no_grad():
        hidden = model.init_hidden(1)
        if isinstance(hidden, tuple):
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:
            hidden = hidden.to(device)
            
        for _ in range(max_length):
            output, hidden = model(seed_tensor, hidden)
            output = output.squeeze(0).div(temperature).exp().cpu()
            output_dist = output[-1].squeeze() 
            top_char = torch.multinomial(output_dist, 1).item()
            generated_char = list(dataset.vocab.keys())[list(dataset.vocab.values()).index(top_char)]
            generated += generated_char
            seed_tensor = torch.tensor([[top_char]], dtype=torch.long).to(device)
    
    return generated

if __name__ == '__main__':
    # 데이터셋 로드
    dataset = ShakespeareDataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 하이퍼파라미터 값 설정
    input_size = len(dataset.vocab)
    hidden_size = 128
    num_layers = 2
    dropout = 0.2

    # 훈련 모델 로드
    model = CharLSTM(input_size, hidden_size, num_layers, input_size, dropout).to(device)
    model.load_state_dict(torch.load('C:/Users/nayun/Downloads/Laguage/model_lstm.pth'))
    
    seed_characters = "The "
    temperature = 1.0
    
    temperatures = [0.5, 1.0, 1.5]
    for temp in temperatures:
        print(f"Temperature: {temp}")
        for _ in range(5):
            sample = generate(model, dataset, seed_characters, temp)
            print(sample)
        print("\n" + "="*50 + "\n")
