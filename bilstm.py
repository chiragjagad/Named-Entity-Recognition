#!/usr/bin/env python
# coding: utf-8

# # Report for Task 1
# ## 1. Load and process data
# - Loading Data: The function load_data reads sentences and their corresponding labels from a file.
# - Preparing Datasets: The loaded data is split into training and development datasets, where each dataset consists of pairs of sentences and their labels.
# - Creating Word-to-Index Mapping: The function create_word_to_index creates a dictionary that assigns a unique index to each word in the training sentences. Special tokens for padding, start, end, and unknown words are also included.
# - Defining Label-to-Index Mapping: The labels_to_index dictionary assigns a unique index to each label used in the dataset.
# - Padding Sequences: The pad_sequences function ensures that all sequences within a batch have the same length. It adds special tokens at the beginning and end of each sequence and pads them to match the length of the longest sequence in the batch.
# - Creating DataLoaders: DataLoaders for both training and development datasets are created. These DataLoaders handle batching, shuffling, and padding of data to prepare it for training and evaluation.
# ## 2. Model Architecture
# - Setting Hyperparameters, Optimizer, Loss Function, and Scheduler: The batch size is 8 and number of epochs is 20. The optimzer is SGD with lr=0.25, momentum=0.9, and weight_decay=0.00005. The loss fucntion is CrossEntropyLoss that ignores padding labels. A scheduler is used to adjust the learning rate based on validation performance. The scheduler used is ReduceLROnPlateau with patience=5 and factor=0.5.
# 
# - Model Architecture: The structure of BiLSTM model consists of embedding, BiLSTM, linear, activation, dropout, and classifier layers where embedding_dim = 100, lstm_hidden_dim = 256, lstm_num_layers = 1, lstm_dropout = 0.33, linear_output_dim = 128 
# 
# ```
# class BiLSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, linear_out_dim, num_labels):
#         super(BiLSTM, self).__init__()
#         # Embedding layer
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         # Bidirectional LSTM layer
#         self.bilstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=lstm_num_layers, bidirectional=True, batch_first=True)
#         # Linear layer
#         self.linear = nn.Linear(lstm_hidden_dim * 2, linear_out_dim)
#         # Activation function
#         self.activation = nn.ELU()
#         # Dropout layer
#         self.dropout = nn.Dropout(lstm_dropout)
#         # Classifier layer
#         self.classifier = nn.Linear(linear_out_dim, num_labels)
# 
#     def forward(self, x):
#         # Embedding layer
#         embedded = self.embedding(x)
#         # Bidirectional LSTM layer
#         lstm_output, _ = self.bilstm(embedded)
#         # Linear layer
#         linear_output = self.linear(lstm_output)
#         # Activation function
#         activation_output = self.activation(linear_output)
#         # Dropout layer
#         dropout_output = self.dropout(activation_output)
#         # Classifier layer
#         logits = self.classifier(dropout_output)
#         return logits
# ```
# 
# - Training and Evaluating the Model: The train_model that takes in training and development data and it trains the model using the training data, evaluates it on the development data, and saves the best-performing model. For each batch, I compute predictions, calculate loss, and update the model's parameters using backpropagation. The function called evaluate_model that assesses the model's performance on the development data. It computes metrics like loss, accuracy, precision, recall, and F1-score.
# After training the model for 20 epochs, I got the following results:
# ```
# Epoch 20/20, Train Loss: 0.0141, Val Loss: 0.1861, Val Accuracy: 0.9645, Val Precision: 0.8122, Val Recall 0.7480, Val F1 Score 0.7602
# ```
# 
# ## 3. ANSWER - Result Analysis
# 
# The precision, recall and F1 score on the dev data as evaluated using eval.py are as follow:
# #### Accuracy:  96.44%; Precision:  86.64%; Recall:  79.03%; F1 score:  82.66%
# 
# ```
# processed 51577 tokens with 5942 phrases; found: 5420 phrases; correct: 4696.
# accuracy:  96.44%; precision:  86.64%; recall:  79.03%; FB1:  82.66
#               LOC: precision:  91.86%; recall:  87.26%; FB1:  89.50  1745
#              MISC: precision:  88.40%; recall:  76.90%; FB1:  82.25  802
#               ORG: precision:  80.52%; recall:  73.97%; FB1:  77.11  1232
#               PER: precision:  84.83%; recall:  75.57%; FB1:  79.93  1641
# ```

# In[1]:


# Import Libraries
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Function to load data from a file
def load_data(file_path):
    sentences = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        sentence_labels = []
        for line in file:
            if line.strip() == '':  # Check for empty line to separate sentences
                if len(sentence) > 0:
                    sentences.append(sentence)
                    labels.append(sentence_labels)
                    sentence = []  # Reset sentence list
                    sentence_labels = []  # Reset label list
            else:
                parts = line.strip().split()
                sentence.append(parts[1])  # Extract word
                sentence_labels.append(parts[2])  # Extract label
    return sentences, labels


# In[3]:


# Load training and development data
train_sentences, train_labels = load_data('data/train')  # Load training data
dev_sentences, dev_labels = load_data('data/dev')  # Load development data


# In[4]:


# Combine sentences and labels into datasets
train_dataset = [(sentence, label) for sentence, label in zip(train_sentences, train_labels)]
dev_dataset = [(sentence, label) for sentence, label in zip(dev_sentences, dev_labels)]


# In[5]:


def create_word_to_index(sentences):
    words_to_index = {}
    index = 0
    # Special tokens for padding, start of sentence, end of sentence, and unknown words
    words_to_index['<PAD>'] = index
    words_to_index['<S>'] = index + 1
    words_to_index['</S>'] = index + 2
    words_to_index['<UNK>'] = index + 3

    index += 4
    # Iterate over each sentence to create word to index mapping
    for sentence in sentences:
        for word in sentence:
            if word not in words_to_index:
                words_to_index[word] = index
                index += 1
    return words_to_index

# Create word to index mapping for training sentences
words_to_index = create_word_to_index(train_sentences)

# Define label to index mapping
labels_to_index = {
    '<PAD>': 0,
    '<S>': 1,
    '</S>': 2,
    'O': 3,
    'I-MISC': 4,
    'I-ORG': 5,
    'I-LOC': 6,
    'I-PER': 7,
    'B-ORG': 8,
    'B-MISC': 9,
    'B-PER': 10,
    'B-LOC': 11,
}


# In[6]:


# def pad_sequences(batch, word_to_index, label_to_index, pad_token='<PAD>', sos_token='<S>', eos_token='</S>', unk_token='<UNK>'):
#     # Calculate the maximum length of sequences in the batch
#     max_length = max([len(seq) + 2 for seq, _ in batch])  # Add 2 for initial and end tokens

#     padded_word_seqs = []
#     padded_label_seqs = []

#     for words, labels in batch:
#         # Padding words with initial and end tokens
#         padded_words = [sos_token] + words + [eos_token]
#         # Convert words to indices, replace unknown words with the index of <UNK>, and pad to max_length
#         padded_words = [word_to_index.get(word, word_to_index[unk_token]) for word in padded_words] + [word_to_index[pad_token]] * (max_length - len(padded_words))
#         padded_word_seqs.append(padded_words)

#         # Padding labels with initial and end tokens
#         padded_labels = [sos_token] + labels + [eos_token]
#         # Convert labels to indices and pad to max_length
#         padded_labels = [label_to_index[label] for label in padded_labels] + [label_to_index[pad_token]] * (max_length - len(padded_labels))
#         padded_label_seqs.append(padded_labels)

#     return torch.tensor(padded_word_seqs), torch.tensor(padded_label_seqs)


# In[7]:


# Create DataLoader for training dataset
# train_loader = DataLoader(
#     train_dataset,  
#     batch_size=8, 
#     collate_fn=lambda data: pad_sequences(data, words_to_index, labels_to_index),  # Padding sequences using pad_sequences function
#     shuffle=True,  # Shuffle the data
# )

# # Create DataLoader for development dataset
# dev_loader = DataLoader(
#     dev_dataset,  
#     batch_size=8,  
#     collate_fn=lambda data: pad_sequences(data, words_to_index, labels_to_index),  # Padding sequences using pad_sequences function
#     shuffle=True,  # Shuffle the data
# )


# In[8]:


# Vocabulary size and number of labels
vocab_size = len(words_to_index)
num_labels = len(labels_to_index) 

# Model hyperparameters
embedding_dim = 100  
lstm_hidden_dim = 256  
lstm_num_layers = 1  
lstm_dropout = 0.33  
linear_output_dim = 128  

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, linear_out_dim, num_labels):
        super(BiLSTM, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=lstm_num_layers, bidirectional=True, batch_first=True)
        # Linear layer
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_out_dim)
        # Activation function
        self.activation = nn.ELU()
        # Dropout layer
        self.dropout = nn.Dropout(lstm_dropout)
        # Classifier layer
        self.classifier = nn.Linear(linear_out_dim, num_labels)

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)
        # Bidirectional LSTM layer
        lstm_output, _ = self.bilstm(embedded)
        # Linear layer
        linear_output = self.linear(lstm_output)
        # Activation function
        activation_output = self.activation(linear_output)
        # Dropout layer
        dropout_output = self.dropout(activation_output)
        # Classifier layer
        logits = self.classifier(dropout_output)
        return logits


# In[9]:


# def train_model(train_loader, dev_loader, model, optimizer, loss_function, scheduler, num_epochs, num_labels):
#     best_f1_score = -1  # Initialize best F1 score
#     best_model = None  # Placeholder for the best model

#     for epoch in range(num_epochs):
#         model.train()  
#         total_loss = 0  
#         total_samples = 0  

#         # Iterate over batches in training data
#         for data in train_loader:
#             word_sequences, label_sequences = data

#             optimizer.zero_grad()  # Zero the gradients

#             predictions = model(word_sequences)  # Forward pass
#             predictions = predictions.view(-1, num_labels)  
#             label_sequences = label_sequences.view(-1)  

#             loss = loss_function(predictions, label_sequences)  # Compute loss
#             loss.backward()  # Backpropagation

#             optimizer.step()  # Update weights

#             # Update total loss and total samples
#             total_loss += loss.item() * word_sequences.size(0)
#             total_samples += word_sequences.size(0)

#         # Calculate average training loss
#         avg_train_loss = total_loss / total_samples

#         # Evaluate model on development data
#         val_loss, val_accuracy, val_precision, val_recall, val_f1_score = evaluate_model(model, dev_loader, loss_function, num_labels)

#         # Update best F1 score and save the best model
#         if val_f1_score > best_f1_score:
#             best_f1_score = val_f1_score
#             best_model = model
#             torch.save(model.state_dict(), "bilstm1.pt")  # Save model parameters

#         scheduler.step(val_loss)  # Adjust learning rate based on validation F1 score

#         # Print epoch statistics
#         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall {val_recall:.4f}, F1_score {val_f1_score:.4f}")

#     return best_model  # Return the best model


# In[10]:


# def evaluate_model(model, dev_loader, loss_function, num_labels):
#     model.eval()  

#     epoch_loss = 0  # Initialize epoch loss
#     y_true = []  # List to store true labels
#     y_pred = []  # List to store predicted labels

#     total_accuracy = 0  # Initialize total accuracy
#     total_amount = 0  
#     total_loss = 0  

#     with torch.no_grad():
#         for data in dev_loader:
#             word_sequences, label_sequences = data

#             predictions = model(word_sequences)  # Forward pass
#             predictions = predictions.view(-1, num_labels)  
#             label_sequences = label_sequences.view(-1)  

#             loss = loss_function(predictions, label_sequences)  # Compute loss
#             total_loss += loss.item()  # Accumulate loss

#             labels = label_sequences.numpy()  
#             predicted_labels = torch.argmax(predictions, dim=1).numpy() 
#             y_true.extend(labels)  # Append true labels
#             y_pred.extend(predicted_labels)  # Append predicted labels

#             mask = labels != 0  # Mask for non-padding labels
#             correct_predictions = (predicted_labels[mask] == labels[mask]).sum()  # Count correct predictions
#             accuracy = correct_predictions / len(labels[mask])  # Compute accuracy

#             total_accuracy += accuracy  
#             epoch_loss += loss  
#             total_amount += 1  

#     # Calculate precision, recall, F1-score using sklearn
#     precision, recall, f1_score, support = precision_recall_fscore_support(
#         y_true,
#         y_pred,
#         average='macro',  # Calculate macro-averaged scores
#         zero_division=0  # Set zero division behavior
#     )

#     # Return average epoch loss, accuracy, precision, recall, and F1-score
#     return (epoch_loss/total_amount), (total_accuracy/total_amount), precision, recall, f1_score


# In[11]:


# Instantiate a BiLSTM model with specified parameters
# model1 = BiLSTM(vocab_size, embedding_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, linear_output_dim, num_labels)


# # In[12]:


# # Define the optimizer with SGD
# optimizer = optim.SGD(model1.parameters(), lr=0.25, momentum=0.9, weight_decay=0.00005)

# # Define the loss function (CrossEntropyLoss) with ignoring padding index
# loss_function = CrossEntropyLoss(ignore_index=labels_to_index['<PAD>'])

# # Define the scheduler to reduce learning rate on plateau
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# # Number of epochs
# num_epochs = 20

# # Train the model using train_loader and evaluate on dev_loader
# train_model(train_loader, dev_loader, model1, optimizer, loss_function, scheduler, num_epochs, num_labels)


# In[13]:


# Instantiate a BiLSTM model with the same parameters as model1
load_model1 = BiLSTM(vocab_size, embedding_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, linear_output_dim, num_labels)

# Load the state dictionary of model1 into load_model1
model1_state_dict = torch.load("bilstm1.pt")
load_model1.load_state_dict(model1_state_dict)


# In[14]:


# Function to predict labels for input text
def predict_labels(model, input_text, words_to_index, index_to_label):

    model.eval()
    # Tokenize and pad input text
    tokenized_input = tokenize_and_pad(input_text, words_to_index)
    input_tensor = torch.tensor([tokenized_input])  
    
    # Perform forward pass
    with torch.no_grad():
        logits = model(input_tensor)
    
    # Get predicted indices and convert them to labels
    predicted_indices = torch.argmax(logits, dim=-1).squeeze().numpy()
    predicted_labels = [index_to_label[idx] for idx in predicted_indices][1:-1]  # Exclude <s> and </s> tokens
    
    return predicted_labels

# Function to tokenize and pad input text
def tokenize_and_pad(text, words_to_index, pad_token='<PAD>', sos_token='<S>', eos_token='</S>', unk_token='<UNK>'):
    tokens = text.split()
    padded_tokens = [sos_token] + tokens + [eos_token]  # Add start and end tokens
    indices = [words_to_index.get(word, words_to_index[unk_token]) for word in padded_tokens]  # Convert tokens to indices
    
    return indices

# Function to create an output file with predicted labels
def create_output_file(model, textFile, outputFile, words_to_index, labels_to_index, test=False):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indices = []
        words = []
        if not test:
            labels = []
        sentence_index = 0  # To keep track of the sentence index

        # Iterate through lines in input file
        for line in input_file:
            if not line.strip():
                if len(words) > 0:
                    index_to_label = {idx: label for label, idx in labels_to_index.items()}

                    # Predict labels for the sentence
                    new_text = " ".join(words)
                    predicted_labels = predict_labels(model, new_text, words_to_index, index_to_label)

                    # Write predictions to output file
                    for i in range(len(indices)):
                        index = indices[i]
                        word = words[i]
                        if not test:
                            label = labels[i]
                        prediction = predicted_labels[i]

                        predictionLine = f"{index} {word} {prediction}\n"
                        output_file.write(predictionLine)
                    
                    output_file.write("\n")
                    sentence_index += 1

                    # Reset lists for the next sentence
                    indices = []
                    words = []
                    if not test:
                        labels = []
            else:
                if test:
                    index, word = line.strip().split()
                else:
                    index, word, label = line.strip().split()
                indices.append(index)
                words.append(word)
                if not test:
                    labels.append(label)


# In[15]:


# Create an dev output file with predicted labels using trained model
create_output_file(load_model1, "data/dev", "dev1.out", words_to_index, labels_to_index)
print("Dev1 Output file created successfully!")

# In[19]:


# Command to run evaluation script eval.py with predicted output file '-p "dev1.out"' and ground truth file '-g "data/dev"'

#get_ipython().system('python eval.py -p "dev1.out" -g "data/dev"')


# In[20]:


# Create an test output file with predicted labels using trained model
create_output_file(load_model1, "data/test", "test1.out", words_to_index, labels_to_index, test=True)
print("Test1 Output file created successfully!")


# In[ ]:




