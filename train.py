import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders, LatexImageDataset
from model import get_model

class CTCLossWrapper(nn.Module):
    def __init__(self):
        super(CTCLossWrapper, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def forward(self, preds, labels, input_lengths, label_lengths):
        preds = preds.log_softmax(2)
        return self.ctc_loss(preds, labels, input_lengths, label_lengths)

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Calculate the correct input lengths based on the output from the encoder
        output_sequence_length = model.encoder.patch_embedding.n_patches
        input_lengths = torch.full(size=(images.size(0),), fill_value=output_sequence_length, dtype=torch.long).to(device)
        label_lengths = torch.sum(labels != 0, dim=1).to(device)  # Compute actual label lengths
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item()}')
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Calculate the correct input lengths based on the output from the encoder
            output_sequence_length = model.encoder.patch_embedding.n_patches
            input_lengths = torch.full(size=(images.size(0),), fill_value=output_sequence_length, dtype=torch.long).to(device)
            label_lengths = torch.sum(labels != 0, dim=1).to(device)  # Compute actual label lengths
            
            outputs = model(images)
            loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    # Parameters
    json_dir = 'latex_db/json'
    img_dir = 'latex_db/output'
    batch_size = 32
    num_epochs = 10
    max_seq_length = 128
    learning_rate = 0.0001
    # subset_size = 1000  # Use only a subset of data

    # Get data loaders
    train_loader = get_dataloaders(json_dir, img_dir, batch_size, max_seq_length)

    # Get the model
    vocab_size = len(train_loader.dataset.char_list)
    model = get_model(vocab_size, max_seq_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss and optimizer
    criterion = CTCLossWrapper()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and evaluating the model
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, train_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Save the model weights
    torch.save(model.state_dict(), 'latex_vit.pth')
    print("Model weights saved to latex_vit.pth")

if __name__ == '__main__':
    main()
