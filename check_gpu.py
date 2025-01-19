import torch

def check_gpu_memory():
    if not torch.cuda.is_available():
        print("No GPU available.")
        return

    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)

    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")

if __name__ == "__main__":
    check_gpu_memory()
