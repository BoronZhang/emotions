import torch
import pickle

def nearMiss(Y:torch.Tensor, stress=2, deletions:int=1) -> torch.Tensor:
    """
    Parameters:
    -----
    `Y`:
    the labels tensor
    `stress`:
    the stress label
    `deletions`: int
    the number of elements to be deleted from each side of the stress label
    """
    twos = (Y == stress).nonzero()
    first, last = twos[0].item(), twos[-1].item()
    return torch.concat((Y[3000:first-deletions], Y[first:last+1], Y[last+1+deletions:-2000]))

if __name__ == "__main__":
    for i in range(2, 18):
        if i == 12:continue
        print(f"Handling {i}")
        with open(rf"WESAD/S{i}/S{i}_n0.pkl", "rb") as file:
            data = pickle.load(file)
        labels:torch.Tensor = data['label'].mode(1).values
        print(f"\tBefore-> 2s: {(labels == 2).sum():,}, not 2s: {(labels != 2).sum():,}")
        labels = nearMiss(labels, deletions=500)
        print(f"\tAfter-> 2s: {(labels == 2).sum():,}, not 2s: {(labels != 2).sum():,}")
        data['label'] = labels
        with open(rf"WESAD/S{i}/S{i}_n1.pkl", "wb") as file:
            pickle.dump(data, file)
    
    
