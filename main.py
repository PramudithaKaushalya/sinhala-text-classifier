import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import TextDataset, collate_fn, build_vocab, tokenize_text
from predict import predict
from model import TextClassifier

print("Hi, Started the program")

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
DATA_PATH = 'data.xlsx'
MODEL_PATH = "trained_model.pth"

def train_and_save_model(model_for_train, vocab_for_train, train_data_for_train_dataset):

    # Create dataset and dataloader
    train_dataset = TextDataset(train_data_for_train_dataset, vocab_for_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Train the model
    model_for_train.train_model(train_dataloader, epochs=EPOCHS)

    # Save the trained model
    torch.save(model_for_train.state_dict(), MODEL_PATH)
    print("Successfully saved the trained model")

def evaluate_model(model_for_evaluate, vocab_for_evaluate, test_data_for_test_dataset):
    test_dataset = TextDataset(test_data_for_test_dataset, vocab_for_evaluate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Evaluate the model
    accuracy = model_for_evaluate.evaluate(test_dataloader)
    print(f"Test Accuracy: {accuracy:.2f}%")


def predict_from_model(model_for_prediction, vocab_for_prediction):

    # Now, let's test prediction with some example input
    input_texts = ["ඔවුන්ගේ සම්බන්ධය රහසකි", "කොළඹ නගරයට ගමන් කිරීම", "ඔවුන් ක්‍රීඩා පිටියේ", "ඔවුන් ධීවර රැකියාවේ නිරත වේ", "ඔහුගේ ආශාව විශේෂය"]
    for input_text in input_texts:
        predicted_class, probabilities = predict(model_for_prediction, input_text, vocab_for_prediction)

        if predicted_class == 1:
            print(f"The text '{input_text}' is classified as **Adult Content** with probability {probabilities[0][1]:.2f}.")
        else:
            print(f"The text '{input_text}' is classified as **Non-Adult Content** with probability {probabilities[0][0]:.2f}.")

if __name__ == "__main__":
    # Load your dataset (Assuming a CSV format with 'text' and 'label' columns)
    data = pd.read_excel(DATA_PATH)  # Replace with your actual file path
    data = data[['text', 'label']]

    # Tokenize the text and prepare the dataset (convert to list of tuples)
    data = [(tokenize_text(row['text']), row['label']) for _, row in data.iterrows()]

    # Split the data into training and testing (70% train, 30% test)
    train_size = int(0.7 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Build vocabulary from the training data
    vocab = build_vocab(train_data, min_freq=1)

    # Define device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(vocab_size=len(vocab), hidden_dim=128, output_dim=2)

    # Ask user for the operation
    operation = input("Enter 'train', 'evaluate', or 'predict': ").strip().lower()

    if operation == "train":
        train_and_save_model(model, vocab, train_data)
    elif operation == "evaluate":
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.to(device)
        evaluate_model(model, vocab, test_data)
    elif operation == "predict":
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.to(device)
        predict_from_model(model, vocab)
    else:
        print("Invalid operation. Please enter 'train', 'evaluate', or 'predict'.")
