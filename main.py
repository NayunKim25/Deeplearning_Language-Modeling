import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import dataset
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt

def plot_losses(trn_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(trn_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

def train(model, trn_loader, device, criterion, optimizer, epoch):
    model.train()
    trn_loss = 0
    for idx, (inputs, targets) in enumerate(trn_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        input_lengths = (inputs != 0).sum(dim=1)
        hidden = model.init_hidden(inputs.size(0))
        optimizer.zero_grad()
        output, hidden = model(inputs, hidden)
        loss = criterion(output.transpose(1, 2), targets)
        loss = (loss * (targets != 0).float()).sum() / input_lengths.sum()
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
    trn_loss /= len(trn_loader)
    print(f'Epoch: {epoch+1}, Train Loss: {trn_loss:.4f}')
    return trn_loss



def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            input_lengths = (inputs != 0).sum(dim=1)
            hidden = model.init_hidden(inputs.size(0))
            output, hidden = model(inputs, hidden)
            loss = criterion(output.transpose(1, 2), targets)
            loss = (loss * (targets != 0).float()).sum()
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f'Validation Loss: {val_loss:.4f}')
    return val_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset.ShakespeareDataset()
    vocab_size = len(data.vocab)
    trn_sampler = SubsetRandomSampler(data.trn_indices)
    val_sampler = SubsetRandomSampler(data.val_indices)
    trn_loader = DataLoader(data, batch_size=32, sampler=trn_sampler)
    val_loader = DataLoader(data, batch_size=32, sampler=val_sampler)

    # 하이퍼파라미터 설정
    hidden_size = 128  # 은닉층 크기 변경
    num_layers = 2
    learning_rate = 0.005  # 학습률 변경

    # Vanilla RNN 학습
    model = CharRNN(vocab_size, hidden_size, num_layers, vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    trn_losses, val_losses = [], []
    for epoch in range(10):
        trn_loss = train(model, trn_loader, device, criterion, optimizer, epoch)
        val_loss = validate(model, val_loader, device, criterion)
        trn_losses.append(trn_loss)
        val_losses.append(val_loss)
    rnn_val_loss = val_losses[-1]

    # RNN 모델 가중치 저장
    torch.save(model.state_dict(), 'C:/Users/nayun/Downloads/Laguage/model_rnn.pth')
    
    # loss 값 시각화
    plot_losses(trn_losses, val_losses)

    # LSTM 학습
    model = CharLSTM(vocab_size, hidden_size, num_layers, vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trn_losses, val_losses = [], []
    for epoch in range(10):
        trn_loss = train(model, trn_loader, device, criterion, optimizer, epoch)
        val_loss = validate(model, val_loader, device, criterion)
        trn_losses.append(trn_loss)
        val_losses.append(val_loss)
    lstm_val_loss = val_losses[-1]

    # lstm 모델 가중치 저장
    torch.save(model.state_dict(), 'C:/Users/nayun/Downloads/Laguage/model_lstm.pth')

    # loss 값 시각화
    plot_losses(trn_losses, val_losses)

    print(f"Vanilla RNN Validation Loss: {rnn_val_loss:.4f}")
    print(f"LSTM Validation Loss: {lstm_val_loss:.4f}")



if __name__ == '__main__':
    main()