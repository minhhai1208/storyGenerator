# Text Generation with LSTM – PyTorch

This project implements a **word-level text generation model** using **LSTM** in PyTorch. The model is trained on a text corpus (e.g., `alice.txt`) to predict the next word in a sequence, enabling the generation of coherent text sequences.

---

## 📘 Overview

The goal of this project is to **generate new text** that mimics the style and structure of the input corpus. Using a recurrent neural network (RNN) architecture, specifically **LSTM (Long Short-Term Memory)**, the model learns sequential patterns in the text and predicts the next word given a sequence of previous words.

Key steps:

1. Tokenize the text and build a **vocabulary dictionary**.
2. Convert the text into sequences of word indices.
3. Prepare batches and sequences for LSTM training.
4. Train the LSTM to predict the next word.
5. Generate new text by sampling from the trained model.

---

## ⚙️ How It Works

1. **Text Preprocessing**  
   - The corpus is read line by line and tokenized into words.  
   - Each unique word is added to a **dictionary** mapping words to indices (`word2idx`) and indices to words (`idx2word`).  
   - Special token `<eos>` is appended to indicate the end of a line.  

2. **Data Preparation**  
   - All words are converted into **integer indices** according to the dictionary.  
   - The sequences are reshaped into **batches** for efficient training.  
   - Each batch has a fixed **sequence length (`timesteps`)**, so LSTM can learn dependencies across words.  

3. **LSTM Model Architecture**  
   - **Embedding Layer**: Converts each word index into a dense vector, capturing semantic meaning.  
   - **LSTM Layer**: Learns sequential patterns (short-term and long-term dependencies) in the word sequences.  
   - **Linear Layer**: Maps the LSTM outputs to the vocabulary size for word prediction.  

4. **Training**  
   - Mini-batches are fed to the model with input sequences and their corresponding target sequences (next word).  
   - **Cross-Entropy Loss** is used to measure prediction error.  
   - **Gradient Clipping** prevents exploding gradients.  
   - The model is optimized using **Adam**.  

5. **Text Generation**  
   - Start with a random word as input.  
   - Predict the next word using the trained LSTM.  
   - Sample words probabilistically from the output distribution.  
   - Feed the predicted word back as input for the next time step.  
   - Repeat to generate a sequence of words forming coherent text.  

---

## 🧩 Tech Stack

| Library | Purpose |
|---------|---------|
| **PyTorch** | Build and train LSTM neural network |
| **NumPy** | Numerical computations |
| **Python** | Scripting and preprocessing |
| **Torch.nn** | Model layers and loss functions |

---

## 🔑 Key Learnings / Skills Demonstrated

- Sequential data processing for text.  
- Building a **custom dictionary and tokenization** pipeline.  
- Implementing **LSTM for sequence prediction** in PyTorch.  
- Applying **embedding layers** to learn word representations.  
- Handling **batching, sequence slicing, and gradient clipping**.  
- Sampling from the trained model to generate realistic text.  

---
