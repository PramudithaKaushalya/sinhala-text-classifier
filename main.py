import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import TextDataset, collate_fn, build_vocab, tokenize_text
from predict import predict
from model import Model

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


def load_trained_model(vocab):
    """Loads the trained model from disk and ensures compatibility."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model with correct vocab size
    model = Model(vocab_size=len(vocab), hidden_dim=128, output_dim=2)

    try:
        # Load model state dict
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()  # Set to evaluation mode
        print("Model successfully loaded from", MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    return model, device


def evaluate_model(vocab_for_evaluate, test_data_for_test_dataset):
    """Evaluates the trained model on the test dataset."""
    model, device = load_trained_model(vocab_for_evaluate)

    test_dataset = TextDataset(test_data_for_test_dataset, vocab_for_evaluate)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Evaluate the model and get all metrics
    accuracy, precision, recall, f1 = model.evaluate(test_dataloader)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")


def predict_from_model(vocab_for_prediction):
    """Runs predictions using the trained model."""
    model, device = load_trained_model(vocab_for_prediction)

    # Sample input texts
    input_texts = [
        "ඔවුන්ගේ සම්බන්ධය රහසකි",
        "කොළඹ නගරයට ගමන් කිරීම",
        "ඔවුන් ක්‍රීඩා පිටියේ",
        "ඔවුන් ධීවර රැකියාවේ නිරත වේ",
        "ඔහුගේ ආශාව විශේෂය"
    ]

    for input_text in input_texts:
        predicted_class, probabilities = predict(model, input_text, vocab_for_prediction)

        if predicted_class == 1:
            print(
                f"The text '{input_text}' is classified as **Adult Content** with probability {probabilities[0][1]:.2f}.")
        else:
            print(
                f"The text '{input_text}' is classified as **Non-Adult Content** with probability {probabilities[0][0]:.2f}.")


if __name__ == "__main__":
    # Load dataset
    data = pd.read_excel(DATA_PATH)
    data = data[['text', 'label']]

    # Tokenize and prepare the dataset
    data = [(tokenize_text(row['text']), row['label']) for _, row in data.iterrows()]

    # Split dataset (70% train, 30% test)
    train_size = int(0.7 * len(data))
    train_data = data[:train_size]

    # Build vocabulary
    vocab = build_vocab(train_data, min_freq=1)

    # Ask user for the operation
    operation = input("Enter 'train', 'evaluate', or 'predict': ").strip().lower()

    if operation == "train":
        train_and_save_model(Model(vocab_size=len(vocab), hidden_dim=128, output_dim=2), vocab, train_data)
    elif operation == "evaluate":
        test_data = data[train_size:]
        evaluate_model(vocab, test_data)
    elif operation == "predict":
        predict_from_model(vocab)
    else:
        print("Invalid operation. Please enter 'train', 'evaluate', or 'predict'.")
