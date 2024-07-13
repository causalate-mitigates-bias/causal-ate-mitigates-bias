from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
import random
from torchtext.data.utils import get_tokenizer

# Define tokenizer
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')



# def test_nn(model, test_loader, criterion, device):
#     model.eval()
#     test_loss = 0.0
#     correct = 0
#     with torch.no_grad():
#         for inputs, targets, _ in tqdm(test_loader, desc="Testing"):  # Ignore lengths
#             inputs, targets = inputs.to(device).long(), targets.to(device).view(-1,
#                                                                                 1)  # Ensure targets have shape (batch_size, 1)
#             outputs = model(inputs)
#             test_loss += criterion(outputs, targets).item()
#             pred = (outputs > 0.5).float()
#             correct += (pred == targets).sum().item()
#     test_loss /= len(test_loader.dataset)
#     accuracy = 100. * correct / len(test_loader.dataset)
#     print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
#     return accuracy


def test_nn(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets, lengths in tqdm(test_loader, desc="Testing"):
            inputs, targets, lengths = inputs.to(device).long(), targets.to(device), lengths.to(device)
            lengths = lengths.cpu().int()  # Ensure lengths are on CPU and of type int
            outputs = model(inputs, lengths)
            test_loss += criterion(outputs, targets).item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total_samples
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)")
    return accuracy

def test_avg_nn(model1, model2, test_loader, criterion, device):
    model1.eval()
    model2.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Testing"):  # Ignore lengths
            # Ensure targets have shape (batch_size, 1)
            inputs, targets = inputs.to(device).long(), targets.to(device).view(-1, 1)
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            # Average the predictions
            outputs = (outputs1 + outputs2) / 2

            test_loss += criterion(outputs, targets).item()
            pred = (outputs > 0.5).float()
            correct += (pred == targets).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return accuracy


def test_avg_nn_with_length(model1, model2, test_loader, criterion, device):
    model1.eval()
    model2.eval()
    test_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets, lengths in tqdm(test_loader, desc="Testing"):  # Include lengths
            inputs, targets, lengths = inputs.to(device).long(), targets.to(device), lengths.to(device)
            lengths = lengths.cpu().int()  # Ensure lengths are on CPU and of type int

            outputs1 = model1(inputs, lengths)
            outputs2 = model2(inputs, lengths)

            # Average the predictions
            outputs = (outputs1 + outputs2) / 2

            test_loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total_samples
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.2f}%)")
    return accuracy


def perturb_sentence_old(original_input, vocab, perturbation_rate=0.5, num_perturbations=25, n_gram_length=5):
    tokens = original_input.split()
    perturbed_sentences = []
    total_tokens = len(tokens)
    num_to_perturb = max(3, int(total_tokens * perturbation_rate))  # Ensure at least three tokens are perturbed
    num_ngrams = num_to_perturb // n_gram_length + 1  # Number of n-grams to perturb

    for _ in range(num_perturbations):
        sentence_as_tokens = tokens[:]
        perturbed_indices = []

        # Generate random starting indices for the n-grams
        for _ in range(num_ngrams):
            start_idx = random.randint(0, total_tokens - n_gram_length)
            perturbed_indices.extend(range(start_idx, start_idx + n_gram_length))

        # Sort and remove duplicates
        perturbed_indices = sorted(set(perturbed_indices))

        for idx in perturbed_indices:
            new_token = vocab[random.randint(0, len(vocab) - 1)]  # Replace with random token from vocab
            sentence_as_tokens[idx] = new_token

        input_after_perturbation = ' '.join(sentence_as_tokens)
        perturbed_sentences.append(input_after_perturbation)

    return perturbed_sentences



def perturb_sentence(original_input, vocab, perturbation_rate=0.5, num_perturbations=25, n_gram_length=5):
    tokens = original_input.split()
    total_tokens = len(tokens)

    if total_tokens < n_gram_length:
        # If the sentence is too short, skip perturbation
        return [original_input]

    perturbed_sentences = []
    num_to_perturb = max(3, int(total_tokens * perturbation_rate))  # Ensure at least three tokens are perturbed
    num_ngrams = num_to_perturb // n_gram_length + 1  # Number of n-grams to perturb

    for _ in range(num_perturbations):
        sentence_as_tokens = tokens[:]
        perturbed_indices = []

        # Generate random starting indices for the n-grams
        for _ in range(num_ngrams):
            start_idx = random.randint(0, total_tokens - n_gram_length)
            perturbed_indices.extend(range(start_idx, start_idx + n_gram_length))

        # Sort and remove duplicates
        perturbed_indices = sorted(set(perturbed_indices))

        perturbed_tokens = [sentence_as_tokens[idx] for idx in perturbed_indices]  # Extract perturbed tokens
        perturbed_part_of_input = ' '.join(perturbed_tokens)
        perturbed_sentences.append(perturbed_part_of_input)

    return perturbed_sentences


def test_ate_nn(model, text_test_loader, criterion, device, vocab,
                perturbation_rate=0.5, num_perturbations=25, n_gram_length=5):
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for text_list, targets in tqdm(text_test_loader, desc="Testing"):
            targets = targets.to(device).view(-1, 1)

            batch_outputs = []
            for sentence in text_list:
                # perturbed_sentences = [sentence]
                perturbed_sentences = perturb_sentence(sentence, vocab.get_itos(),
                                                       perturbation_rate, num_perturbations, n_gram_length)

                perturbed_scores = []
                for input_after_perturbation in perturbed_sentences:
                    tokenized_input = torch.tensor(vocab(tokenizer(input_after_perturbation)), dtype=torch.int64).to(
                        device).unsqueeze(0)
                    score = model(tokenized_input).item()
                    perturbed_scores.append(score)

                if not perturbed_scores:
                    continue

                max_score = max(perturbed_scores)
                min_score = min(perturbed_scores)
                final_score = min_score if (1 - min_score) > max_score else max_score
                # perturbed_scores_tensor = torch.tensor(perturbed_scores, dtype=torch.float32).to(device)
                # final_score = torch.mean(perturbed_scores_tensor).item()
                batch_outputs.append(final_score)

            if not batch_outputs:
                continue

            batch_outputs = torch.tensor(batch_outputs).to(device).view(-1, 1)
            test_loss += criterion(batch_outputs, targets).item()
            pred = (batch_outputs > 0.5).float()
            correct += (pred == targets).sum().item()

    test_loss /= len(text_test_loader.dataset)
    accuracy = 100. * correct / len(text_test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(text_test_loader.dataset)} ({accuracy:.2f}%)")
    return accuracy


def test_sklearn_model(model, vectorizer, test_loader, device):
    texts, labels = [], []
    for batch in test_loader:
        if len(batch) == 2:
            text_list, label_list = batch
        else:
            text_list, label_list, _ = batch  # Ignore lengths if present
        texts.extend(text_list)
        labels.extend(label_list)

    X_test = vectorizer.transform(texts)
    y_test = torch.tensor(labels, dtype=torch.float32).numpy()

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy * 100
