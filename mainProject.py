import torch
import os
import torch.nn as nn
import numpy
from torch.nn.utils import clip_grad_norm_


class Dictionary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)
    
class textProcess(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size):
        with open(path, "r") as f:
            tokens  = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        rep_tensor = torch.LongTensor(tokens)
        index = 0
        with open(path, "r") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    rep_tensor[index] = self.dictionary.word2idx[word]
                    index += 1
        
        
        num_batches = rep_tensor.shape[0] // batch_size
        print(num_batches)
        rep_tensor = rep_tensor[:num_batches*batch_size]
        rep_tensor = rep_tensor.view(batch_size, -1)
        return rep_tensor 
    
class textGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(textGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_size):
        # perform word embedding
        x = self.embed(x)

        # the shape of out is (batch_size*timestemps*hidden_size)
        # h is the hidden state (short-term memory) ,c is long-term memory
        out, (h,c) = self.lstm(x, hidden_size)
        print(h.shape)
        print(c.shape)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))

        out = self.linear(out)
        return out, (h,c )

def detach(states):
    return [state.detach() for state in states]   

     
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 20
batch_size = 15
timesteps = 30
learning_rate = 0.002


corpus = textProcess()
rep_tensor = corpus.get_data("D:\\alice.txt", batch_size=batch_size)
vocab_size = len(corpus.dictionary)
print(rep_tensor.shape)
model = textGenerator(vocab_size, embed_size, hidden_size, num_layers)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size),
              torch.zeros(num_layers, batch_size, hidden_size))
    num = 0
    for i in range(0, rep_tensor.size(1) - timesteps, timesteps):
        # Get mini-batch inputs and targets
        inputs = rep_tensor[:, i:i+timesteps]  
        targets = rep_tensor[:, (i+1):(i+1)+timesteps]
        
        outputs,_ = model(inputs, states)
        loss = loss_fn(outputs, targets.reshape(-1))

        model.zero_grad()
        loss.backward()
        #Perform Gradient Clipping. clip_value (float or int) is the maximum allowed value of the gradients 
        #The gradients are clipped in the range [-clip_value, clip_value]. This is to prevent the exploding gradient problem
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        _, predicted = torch.max(outputs, dim=1)  # Get predicted class indices
        predicted = predicted.view(-1)  # Flatten the predictions
        targets_flat = targets.reshape(-1)  # Flatten the targets

        correct = (predicted == targets_flat).sum().item()  # Count correct predictions
        total = targets_flat.size(0)  # Total number of predictions

        accuracy = correct / total  # Calculate accuracy
              
        step = (i+1) // timesteps
        if step % 100 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                   .format(epoch+1, num_epochs, loss.item(), accuracy*100))
        num += 1
        print(num)    
with torch.no_grad():
    with open('D:\\results.txt', 'w') as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size),
                 torch.zeros(num_layers, 1, hidden_size))
        # Select one word id randomly and convert it to shape (1,1)
        input = torch.randint(0,vocab_size, (1,)).long().unsqueeze(1)

        for i in range(500):
            output, _ = model(input, state)
            print(output.shape)
            # Sample a word id from the exponential of the output 
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
            print(word_id)
            # Replace the input with sampled word id for the next time step
            input.fill_(word_id)

            # Write the results to file
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)
            
            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, 500, 'results.txt'))