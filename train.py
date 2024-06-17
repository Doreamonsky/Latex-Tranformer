import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model

def train(model, loader, criterion, optimizer, device):
    model.train()
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, labels[:, :-1])
        loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item()}')

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, labels[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # Parameters
    json_dir = 'latex_db/json'
    img_dir = 'latex_db/output'
    batch_size = 64
    num_epochs = 10
    max_seq_length = 128
    learning_rate = 0.001

    # Get data loaders
    train_loader = get_dataloaders(json_dir, img_dir, batch_size, max_seq_length)

    # Get the model
    vocab_size = len(train_loader.dataset.char_list)
    model = get_model(vocab_size, max_seq_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and evaluating the model
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, train_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

    # Save the model weights
    torch.save(model.state_dict(), 'latex_vit.pth')
    print("Model weights saved to latex_vit.pth")

    # Example of loading and using the model for prediction
    model.load_state_dict(torch.load('latex_vit.pth'))
    model.eval()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images, labels[:, :-1])
        predicted_labels = outputs.argmax(dim=-1)
        for i in range(len(images)):
            encoded_label = predicted_labels[:, i].cpu().numpy()
            decoded_label = train_loader.dataset.decode_from_labels(encoded_label)
            print(f"Predicted: {decoded_label}")
        break

if __name__ == '__main__':
    main()
