import torch 



_default_device = None  

def set_default_device(device):
    global _default_device
    _default_device = torch.device(device)

def get_default_device():
    return _default_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")