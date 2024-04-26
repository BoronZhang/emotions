import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class NeuralNet1(nn.Module):
    def __init__(self, length, in_channels=1) -> None:
        """
        Parameters:
        -----
        `length`: the window size
        `in_channels`: number of input channels\\
        for most cases is 1, for ACC is 3
        """
        super(NeuralNet1, self).__init__()
        kernel_size, stride, maxPoold = 64, 8, 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxPoold)
        )
        length = (length - kernel_size) // stride + 1
        length = (length - maxPoold) // maxPoold + 1
        
        kernel_size, stride, maxPoold = 32, 4, 2
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, out_channels=128, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=maxPoold)
        )
        length = (length - kernel_size) // stride + 1
        length = (length - maxPoold) // maxPoold + 1
        self.fc = nn.Sequential(
            nn.Linear(128 * length, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    
    def forward(self, x:torch.Tensor):
        x = x.unsqueeze(1)
        # print(f'after reshape: {x.shape}')
        x = self.conv1(x)
        # print(f'after conv1: {x.shape}')
        x = self.conv2(x)
        # print(f'after conv2: {x.shape}')
        x = x.flatten(1)
        # print(f'after flatten: {x.shape}')
        x = self.fc(x)
        # print(f'after fc: {x.shape}')
        return x
    
class Trainer:
    def __init__(self, inp, target, window_size, in_channels=1, device='cpu', lr=0.001, num_epochs=10, batch_size=32) -> None:
        self.model = NeuralNet1(window_size, in_channels=in_channels).to(device)
        # Define loss function and optimizer
        self.criterion = torch.nn.BCEWithLogitsLoss()  # Changed loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.inptGen = self.process_input(inp, window_size=window_size, batch_size=batch_size)
        self.trgtGen = self.process_input(target, window_size=window_size, batch_size=batch_size)
    
    def process_input(data, window_size, batch_size, stride=1):
        """
        Parameters
        -----
        `data`
        `window_size`
        `batch_size`
        `stride`

        Returns
        -----
        yields a numpy array of size (`batch_size`, `window_size`, `in_channels`, `events_in_a_second`)
        where `in_channels` is the length of each of the rows of the `data`
        """
        for i in range(0, len(data), stride * batch_size):
            yield np.array([data[j * stride:j*stride+window_size] for j in range(i, i+batch_size)])


        

    def go(self):
        for epoch in range(self.num_epochs):
            for i in range(0, len(self.input), self.batch_size):
                batch_input = next(self.inptGen)
                batch_target = next(self.trgtGen).unsqueeze(1)  # Add dimension for BCEWithLogitsLoss
                
                # Forward pass
                outputs = self.model(batch_input)
                
                # print(f"Output: {outputs}:{outputs.shape}\nExpected: {batch_target}:{batch_target.shape}")
                loss = self.criterion(outputs, batch_target)
                # print(f"loss: {loss}")
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # Print progress after each epoch
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')

