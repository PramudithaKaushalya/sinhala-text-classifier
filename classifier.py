import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim):
        """
        Initializes the text classification model.

        Parameters:
            vocab_size (int): Size of the vocabulary.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Number of output classes (2: adult/non-adult).
        """
        super(TextClassifier, self).__init__()

        # Define layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
            x (tensor): Input tensor of tokenized text.

        Returns:
            tensor: Output logits for each class.
        """
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        output = self.fc(rnn_out[:, -1, :])  # Use the last RNN output
        return output

    def train_model(self, train_dataloader, epochs=5, lr=0.001):
        """
        Train the model.

        Parameters:
            train_dataloader (DataLoader): DataLoader for training data.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for optimization.
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            running_loss = 0.0
            for texts, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = self(texts)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_dataloader)}")

    def evaluate(self, test_dataloader):
        """
        Evaluate the model on test data.

        Parameters:
            test_dataloader (DataLoader): DataLoader for test data.

        Returns:
            accuracy (float): Accuracy of the model on the test data.
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in test_dataloader:
                outputs = self(texts)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy