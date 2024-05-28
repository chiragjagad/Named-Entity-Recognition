#!/usr/bin/env python
# coding: utf-8

# # Report for Task 3 - Bonus
# ## 1. Load and process data
# - Loading Data: The function load_data reads sentences and their corresponding labels from a file.
# - Preparing Datasets: The loaded data is split into training and development datasets, where each dataset consists of pairs of sentences and their labels.
# - GloVE Embeddings: A dictionary named glove_word_to_idx is created to establish a mapping between GloVe words and their respective indices. Predefined indices are assigned to special tokens such as pad, s, /s, and unk within this dictionary.The GloVe embeddings file (glove.6B.100d.gz) is accessed and its contents are parsed. Each line of the file represents a word followed by its corresponding embedding values. The words are assigned indices and both the words and their embeddings are stored in separate lists. Embeddings for the special tokens, including pad =0 , s = 0, /s = 0 are defined. The unk token's embedding is computed as the mean of all GloVe embeddings, providing a representative representation for unknown words. The embeddings for the special tokens are inserted at the start of the embeddings array.
# - Char-to-Index Mapping: The function create_char_mapping that generates a character-to-index mapping based on characters present in the input dataset. It iterates through each word in the dataset and assigns a unique index to each character encountered. The special tokens for padding and unknown characters are also included in the dictionary. The resulting mapping is stored in the char_to_index dictionary.
# - Defining Label-to-Index Mapping: The labels_to_index dictionary assigns a unique index to each label used in the dataset.
# - Padding Sequences: The pad_sequences function ensures that all sequences within a batch have the same length. It adds special tokens at the beginning and end of each sequence and pads them to match the length of the longest sequence in the batch.
# - Handling case-sensitivity: I added an extra array to each word to extract the additional features of cases. I convert each word in the sequence to a list of feature values, which includes whether the word is in title case, uppercase, or lowercase. These feature values are represented as binary indicators, with 1.0 indicating the presence of the feature and 0.0 indicating absence. Afterwards, I pad the features to match the maximum sequence length. Any remaining positions are filled with a list of zeros representing the absence of features.
# - Handling character-level features: The characters (indices) within words are used as one of inputs for neural network along with words (indices) and labels (indices). Each character in every word is replaced with its corresponding index. If a character is not found in the mapping, it is replaced with the index of a special token unk representing unknown characters. Subsequently, the character indices for each word are padded to match the maximum word length. Padding tokens are then inserted at the beginning and end of the list of character indices to denote the start and end of each word. The character sequences are padded with additional padding tokens to align with the maximum sequence length.
# - Creating DataLoaders: DataLoaders for both training and development datasets are created. These DataLoaders handle batching, shuffling, and padding of data to prepare it for training and evaluation.
# ## 2. Model Architecture
# - Setting Hyperparameters, Optimizer, Loss Function, and Scheduler: The batch size is 8 and number of epochs is 20. The optimzer is SGD with lr=0.25, momentum=0.9, and weight_decay=0.00005. The loss fucntion is CrossEntropyLoss that ignores padding labels. A scheduler is used to adjust the learning rate based on validation performance. The scheduler used is ReduceLROnPlateau with patience=5 and factor=0.5.
# 
# - Model Architecture: The structure of BiLSTM model consists of embedding, character embedding layer, character CNN layer (kernel_size=3, padding=1), BiLSTM, linear, activation, dropout, and classifier layers where char_embedding_dim = 30, embedding_dim = 100, lstm_hidden_dim = 256, lstm_num_layers = 1, lstm_dropout = 0.33, linear_output_dim = 128. The structure of the BiLSTM model is similar to task 1. But for the embedding layer uses pre-trained GloVe embeddings. It initializes the embedding layer with the GloVe embeddings stored as a NumPy array and freezes the weights (freeze=True) to retain the pre-trained embeddings. The forward pass is similar expect the start. I first embed input sequences using an embedding layer. Then, I process character features with a separate embedding layer followed by a CNN layer to capture patterns. After applying ReLU activation and max-pooling, I reshape the tensor to match the original batch size and sequence length. Next, I concatenate the embeddings, additional features, and processed character features. Finally, I feed the concatenated tensor into a bidirectional LSTM layer to capture contextual information (including case sensitivity and character-level features) from from both forward and backward directions in the sequence.
# 
# ```
# # Create an embedding layer from pretrained GloVe embeddings
# glove_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(glove_embeddings).float(), freeze=True, padding_idx=0)
# 
# # Define the BiLSTM model
# class BiLSTM_GloVE(nn.Module):
#     def __init__(self, embedding_dim, char_embedding_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, linear_out_dim, num_labels, char_vocab_size):
#         super(BiLSTM_GloVE, self).__init__()
#         # Embedding layer
#         self.embedding = glove_embedding_layer
#         
#         self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
#         self.char_cnn = nn.Conv1d(char_embedding_dim, embedding_dim, kernel_size=3, padding=1)
#         # Bidirectional LSTM layer
#         self.bilstm = nn.LSTM((embedding_dim*2)+3, lstm_hidden_dim, num_layers=lstm_num_layers, bidirectional=True, batch_first=True)
#         # Linear layer
#         self.linear = nn.Linear(lstm_hidden_dim * 2, linear_out_dim)
#         # Activation function
#         self.activation = nn.ELU()
#         #Dropout layer
#         self.dropout = nn.Dropout(lstm_dropout)
#         # Classifier layer
#         self.classifier = nn.Linear(linear_out_dim, num_labels)
# 
#     def forward(self, x, x_features, char_features):
#         # Embedding layer
#         embedded = self.embedding(x)
#         
#         # Process character features
#         char_features = self.char_embedding(char_features)
#         batch_size, max_seq_len, max_word_len, _ = char_features.shape
#         char_features_reshaped = char_features.view(batch_size * max_seq_len, max_word_len, -1).permute(0, 2, 1)
#         cnn_char_features = self.char_cnn(char_features_reshaped)
#         cnn_char_features = nn.functional.relu(cnn_char_features)
#         cnn_char_features, _ = torch.max(cnn_char_features, dim=-1)
#         cnn_char_features = cnn_char_features.view(batch_size, max_seq_len, -1)
#         
#         # Concatenate the embedded, characters, and features tensors
#         embedded_with_features = torch.cat((embedded, x_features, cnn_char_features), dim= 2)
#         # Bidirectional LSTM layer
#         lstm_output, _ = self.bilstm(embedded_with_features)
#         # Linear layer
#         linear_output = self.linear(lstm_output)
#         # Activation function
#         activation_output = self.activation(linear_output)
#         #Dropout layer
#         dropout_output = self.dropout(activation_output)
#         # Classifier layer
#         logits = self.classifier(dropout_output)
#         return logits
# 
# ```
# - Training and Evaluating the Model: The train_model that takes in training and development data and it trains the model using the training data, evaluates it on the development data, and saves the best-performing model. For each batch, I compute predictions, calculate loss, and update the model's parameters using backpropagation. The function called evaluate_model that assesses the model's performance on the development data. It computes metrics like loss, accuracy, precision, recall, and F1-score.
# After training the model for 20 epochs, I got the following results:
# ```
# Epoch 20/20, Train Loss: 0.0220, Val Loss: 0.0620, Val Accuracy: 0.9846, Val Precision: 0.7863, Val Recall 0.8350, Val F1 Score 0.7731
# ```
# ## 3. Result Analysis
# The precision, recall and F1 score on the dev data as evaluated using eval.py are as follow:
# #### Accuracy:  98.38%; Precision:  89.14%; Recall:  91.03%; F1 Score  90.07
# ```
# processed 51577 tokens with 5942 phrases; found: 6068 phrases; correct: 5409.
# accuracy:  98.38%; precision:  89.14%; recall:  91.03%; FB1:  90.07
#               LOC: precision:  91.13%; recall:  96.79%; FB1:  93.88  1951
#              MISC: precision:  84.73%; recall:  82.43%; FB1:  83.56  897
#               ORG: precision:  85.71%; recall:  84.56%; FB1:  85.14  1323
#               PER: precision:  91.57%; recall:  94.30%; FB1:  92.91  1897
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
import gzip
import numpy as np
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


# Dictionary to store GloVe word to index mapping
glove_word_to_idx = {}

glove_vocab, glove_embeddings = [], []

# Special tokens for padding, start of sentence, end of sentence, and unknown words
glove_word_to_idx['<pad>'] = 0
glove_word_to_idx['<s>'] = 1
glove_word_to_idx['</s>'] = 2
glove_word_to_idx['<unk>'] = 3

# Open GloVe embeddings file
with gzip.open('glove.6B.100d.gz', 'rt') as f:
    all_file_embeddings = f.read().strip().split('\n')

# Extract words and embeddings from GloVe file
for i in range(len(all_file_embeddings)):
    glove_word = all_file_embeddings[i].split(' ')[0]  # Extract word
    glove_emb = [float(x) for x in all_file_embeddings[i].split(' ')[1:]]  # Extract embedding
    glove_word_to_idx[glove_word] = i + 4  # Assign index to word
    glove_vocab.append(glove_word)  # Add word to vocabulary
    glove_embeddings.append(glove_emb)  # Add embedding to list

# Convert vocabulary and embeddings to numpy arrays
glove_vocab = np.array(glove_vocab)
glove_embeddings = np.array(glove_embeddings)

# Add special tokens to vocabulary numpy array
glove_vocab = np.insert(glove_vocab, 0, '<pad>')
glove_vocab = np.insert(glove_vocab, 1, '<s>')
glove_vocab = np.insert(glove_vocab, 2, '</s>')
glove_vocab = np.insert(glove_vocab, 3, '<unk>')

# Define embeddings for special tokens
pad_emb = np.zeros((1, glove_embeddings.shape[1]))  # Embedding for '<pad>' token
start_emb = np.zeros((1, glove_embeddings.shape[1]))  # Embedding for '<s>' token
end_emb = np.zeros((1, glove_embeddings.shape[1]))  # Embedding for '</s>' token
unk_emb = np.mean(glove_embeddings, axis=0, keepdims=True)  # Embedding for '<unk>' token

# Insert embeddings for special tokens at the beginning of embeddings numpy array
glove_embeddings = np.vstack((pad_emb, start_emb, end_emb, unk_emb, glove_embeddings))


# In[6]:


# Define label to index mapping
labels_to_index = {
    '<pad>': 0,
    '<s>': 1,
    '</s>': 2,
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


# In[7]:


# Function to create a character to index mapping
def create_char_mapping(data):
    char_to_index = {}
    char_to_index['<pad>'] = 0  # Padding character
    char_to_index['<unk>'] = 1  # Unknown character
    
    # Iterate through all words in the dataset
    for sentence, _ in data:
        for word in sentence:
            for char in word:
                if char not in char_to_index:
                    char_to_index[char] = len(char_to_index)
    
    return char_to_index

# Create character to index mapping for training data
char_to_index = create_char_mapping(train_dataset)


# In[8]:


# def pad_word_chars(chars, max_word_len, pad_idx):
#     return chars + [pad_idx] * (max_word_len - len(chars))

# def pad_sequences(batch, word_to_index, char_to_index, label_to_index, pad_token='<pad>', sos_token='<s>', eos_token='</s>', unk_token='<unk>'):
#     # Calculate the maximum length of sequences in the batch
#     max_length = max([len(seq) + 2 for seq, _ in batch])  # Add 2 for initial and end tokens
#     # Calculate the maximum length of words in the sequence
#     max_word_len = max([len(word) for seq, _ in batch for word in seq])

#     padded_word_seqs = []
#     padded_char_seqs = []
#     padded_label_seqs = []
#     padded_features_seqs = []
    
#     for words, labels in batch:
#         # Padding words with initial and end tokens
#         padded_words = [sos_token] + words + [eos_token]
#         # Convert words to indices, replace unknown words with the index of <UNK>, and pad to max_length
#         padded_words = [word_to_index.get(word.lower(), word_to_index[unk_token]) for word in padded_words] + [word_to_index[pad_token]] * (max_length - len(padded_words))
#         padded_word_seqs.append(padded_words)

#         # Pad characters for each word
#         padded_chars = [[char_to_index.get(char, char_to_index['<unk>']) for char in word] for word in words]
#         # Pad each word's character indices to match the maximum word length
#         padded_chars = [pad_word_chars(chars, max_word_len, char_to_index[pad_token]) for chars in padded_chars]
#         # Insert padding tokens at the beginning and end of the list of character indices
#         padded_chars.insert(0, [char_to_index[pad_token]] * max_word_len)
#         padded_chars.append([char_to_index[pad_token]] * max_word_len)
#         # Pad the character sequences with padding tokens to match the maximum sequence length   
#         padded_chars += [[char_to_index[pad_token]] * max_word_len] * (max_length - len(padded_chars))
#         padded_char_seqs.append(padded_chars)

#         # Padding labels with initial and end tokens
#         padded_labels = [sos_token] + labels + [eos_token]
#         # Convert labels to indices and pad to max_length
#         padded_labels = [label_to_index[label] for label in padded_labels] + [label_to_index[pad_token]] * (max_length - len(padded_labels))
#         padded_label_seqs.append(padded_labels)
        
#         # Padding features with initial and end tokens
#         padded_features = [sos_token] + words + [eos_token]
#         # Convert each word to a list of feature values (title case, uppercase, lowercase)
#         padded_features = [[float(str(word).istitle()), float(str(word).isupper()), float(str(word).islower())] for word in padded_features]
#         # Pad the features match the maximum length
#         padded_features += [[0.0, 0.0, 0.0]] * (max_length - len(padded_features))
#         padded_features_seqs.append(padded_features)
    
#     # Convert padded sequences to torch tensors
#     return (
#         torch.tensor(padded_word_seqs),
#         torch.tensor(padded_char_seqs),
#         torch.tensor(padded_features_seqs),
#         torch.tensor(padded_label_seqs)
#     )


# In[9]:


# Create DataLoader for training dataset
# train_loader = DataLoader(
#     train_dataset,  
#     batch_size=8, 
#     collate_fn=lambda data: pad_sequences(data, glove_word_to_idx, char_to_index, labels_to_index),  # Padding sequences using pad_sequences function
#     shuffle=True,  # Shuffle the data
# )

# # Create DataLoader for development dataset
# dev_loader = DataLoader(
#     dev_dataset,  
#     batch_size=8,  
#     collate_fn=lambda data: pad_sequences(data, glove_word_to_idx, char_to_index, labels_to_index),  # Padding sequences using pad_sequences function
#     shuffle=True,  # Shuffle the data
# )


# In[10]:


# # Check the shape of data
# for batch_idx, (data, chars, features, target) in enumerate(train_loader):
#     print(f"Batch {batch_idx}:")
#     print("Data shape:", data.shape)  
#     print("Chars shape:", chars.shape)  
#     print("Features shape:", features.shape)
#     print("Labels shape:", target.shape)
#     break


# In[22]:


# Vocabulary size and number of labels
vocab_size = len(glove_word_to_idx)
num_labels = len(labels_to_index) 
char_vocab_size = len(char_to_index)

# Model hyperparameters
char_embedding_dim = 30
embedding_dim = 100  
lstm_hidden_dim = 256  
lstm_num_layers = 1  
lstm_dropout = 0.33  
linear_output_dim = 128

# Create an embedding layer from pretrained GloVe embeddings
glove_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(glove_embeddings).float(), freeze=True, padding_idx=0)

# Define the BiLSTM model
class BiLSTM_GloVE(nn.Module):
    def __init__(self, embedding_dim, char_embedding_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, linear_out_dim, num_labels, char_vocab_size):
        super(BiLSTM_GloVE, self).__init__()
        # Embedding layer
        self.embedding = glove_embedding_layer
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_cnn = nn.Conv1d(char_embedding_dim, embedding_dim, kernel_size=3, padding=1)
        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM((embedding_dim*2)+3, lstm_hidden_dim, num_layers=lstm_num_layers, bidirectional=True, batch_first=True)
        # Linear layer
        self.linear = nn.Linear(lstm_hidden_dim * 2, linear_out_dim)
        # Activation function
        self.activation = nn.ELU()
        #Dropout layer
        self.dropout = nn.Dropout(lstm_dropout)
        # Classifier layer
        self.classifier = nn.Linear(linear_out_dim, num_labels)

    def forward(self, x, x_features, char_features):
        # Embedding layer
        embedded = self.embedding(x)
        
        # Process character features
        char_features = self.char_embedding(char_features)
        batch_size, max_seq_len, max_word_len, _ = char_features.shape
        char_features_reshaped = char_features.view(batch_size * max_seq_len, max_word_len, -1).permute(0, 2, 1)
        cnn_char_features = self.char_cnn(char_features_reshaped)
        cnn_char_features = nn.functional.relu(cnn_char_features)
        cnn_char_features, _ = torch.max(cnn_char_features, dim=-1)
        cnn_char_features = cnn_char_features.view(batch_size, max_seq_len, -1)
        
        # Concatenate the embedded, characters, and features tensors
        embedded_with_features = torch.cat((embedded, x_features, cnn_char_features), dim= 2)
        # Bidirectional LSTM layer
        lstm_output, _ = self.bilstm(embedded_with_features)
        # Linear layer
        linear_output = self.linear(lstm_output)
        # Activation function
        activation_output = self.activation(linear_output)
        #Dropout layer
        dropout_output = self.dropout(activation_output)
        # Classifier layer
        logits = self.classifier(dropout_output)
        return logits


# In[23]:


# def train_model(train_loader, dev_loader, model, optimizer, loss_function, scheduler, num_epochs, num_labels):
#     best_f1_score = -1  # Initialize best F1 score
#     best_model = None  # Placeholder for the best model

#     for epoch in range(num_epochs):
#         model.train()  
#         total_loss = 0  
#         total_samples = 0  

#         # Iterate over batches in training data
#         for data in train_loader:
#             word_sequences, char_sequences, feature_sequences, label_sequences = data

#             optimizer.zero_grad()  # Zero the gradients
#             predictions = model(word_sequences, feature_sequences, char_sequences)  # Forward pass
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
#             torch.save(model.state_dict(), "bilstm3.pt")  # Save model parameters

#         scheduler.step(val_loss)  # Adjust learning rate based on validation F1 score

#         # Print epoch statistics
#         print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall {val_recall:.4f}, F1_score {val_f1_score:.4f}")

#     return best_model  # Return the best model


# # In[24]:


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
#             word_sequences, char_sequences, feature_sequences, label_sequences = data
            
#             predictions = model(word_sequences, feature_sequences, char_sequences)  # Forward pass
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


# # In[25]:


# # Instantiate a BiLSTM model with specified parameters
# model3 = BiLSTM_GloVE(embedding_dim, char_embedding_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, linear_output_dim, num_labels, char_vocab_size)


# # In[26]:


# # Define the optimizer with SGD
# optimizer = optim.SGD(model3.parameters(), lr=0.25, momentum=0.9, weight_decay=0.00005)

# # Define the loss function (CrossEntropyLoss) with ignoring padding index
# loss_function = CrossEntropyLoss(ignore_index=labels_to_index['<pad>'])

# # Define the scheduler to reduce learning rate on plateau
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# # Number of epochs
# num_epochs = 20

# # Train the model using train_loader and evaluate on dev_loader
# train_model(train_loader, dev_loader, model3, optimizer, loss_function, scheduler, num_epochs, num_labels)


# In[27]:


# Instantiate a BiLSTM model with the same parameters as model3
load_model3 = BiLSTM_GloVE(embedding_dim, char_embedding_dim, lstm_hidden_dim, lstm_num_layers, lstm_dropout, linear_output_dim, num_labels, char_vocab_size)
# Load the state dictionary of model1 into load_model3
model3_state_dict = torch.load("bilstm3.pt")

load_model3.load_state_dict(model3_state_dict)


# In[28]:


# Function to predict labels for input text
def predict_labels(model, input_text, words_to_index, index_to_label, char_to_index):

    model.eval()
    # Tokenize and pad input text and features
    tokenized_input, features, char_features = tokenize_and_pad(input_text, words_to_index, char_to_index)
    input_tensor = torch.tensor([tokenized_input])  
    features_tensor = torch.tensor([features])  
    char_features_tensor = torch.tensor([char_features])  

    # Perform forward pass
    with torch.no_grad():
        logits = model(input_tensor, features_tensor, char_features_tensor)
    
    # Get predicted indices and convert them to labels
    predicted_indices = torch.argmax(logits, dim=-1).squeeze().numpy()
    predicted_labels = [index_to_label[idx] for idx in predicted_indices][1:-1]  # Exclude <s> and </s> tokens
    
    return predicted_labels

# Function to tokenize and pad input text
def tokenize_and_pad(text, words_to_index, char_to_index, pad_token='<pad>', sos_token='<s>', eos_token='</s>', unk_token='<unk>'):
    tokens = text.split()
    padded_tokens = [sos_token] + tokens + [eos_token]  # Add start and end tokens
    indices = [words_to_index.get(word.lower(), words_to_index[unk_token]) for word in padded_tokens]  # Convert tokens to indices
    
    features = [[float(str(word).istitle()), float(str(word).isupper()), float(str(word).islower())] for word in padded_tokens] # Get features
    
    # Get char indices and pad 
    char_indices = [[char_to_index.get(char, char_to_index[unk_token]) for char in word] for word in tokens]
    max_word_len = max([len(word_chars) for word_chars in char_indices]) + 2
    char_indices = [[char_to_index[pad_token]] * max_word_len] + char_indices + [[char_to_index[pad_token]] * max_word_len]
    char_indices_padded = [word_chars + [char_to_index[pad_token]] * (max_word_len - len(word_chars)) for word_chars in char_indices]
    
    return indices, features, char_indices_padded

# Function to create an output file with predicted labels
def create_output_file(model, textFile, outputFile, words_to_index, labels_to_index, char_to_index, test=False):
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
                    predicted_labels = predict_labels(model, new_text, words_to_index, index_to_label, char_to_index)

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


# In[29]:


# Create an dev output file with predicted labels using trained model
# create_output_file(load_model3, "data/dev", "pred_dev.out", glove_word_to_idx, labels_to_index, char_to_index)


# In[30]:


# Command to run evaluation script eval.py with predicted output file '-p "dev1.out"' and ground truth file '-g "data/dev"'
# Task 3 does not need dev file
# get_ipython().system('python eval.py -p "pred_dev.out" -g "data/dev"')


# In[31]:


# Create an test output file with predicted labels using trained model
create_output_file(load_model3, "data/test", "pred.out", glove_word_to_idx, labels_to_index, char_to_index, test=True)
print("Test Bonus Output file created successfully!")


# In[ ]:




