import torch
print("CUDA disponible:", torch.cuda.is_available())
print("Dispositivo actual:", torch.cuda.current_device())
print("Nombre de GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
