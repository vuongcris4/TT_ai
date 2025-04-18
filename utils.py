from torch.utils.data import random_split

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

def train_val_split(dataset):
    # 80% train, 20% test
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    return train_dataset, val_dataset