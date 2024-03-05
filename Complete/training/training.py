import tqdm
import torch
import constants as const


def get_loss(model, image, t):
    pass


def train(dataloader, model, criterion, num_epochs, optimizer, device):
    """
    Trains the model for a given number of epochs
    Params:
    train_loader: DataLoader - the training data
    val_loader: DataLoader - the validation data
    model: nn.Module - the model to be trained
    criterion: function - loss function that takes in the model's output and the target
    num_epochs: int - the number of epochs to train the model
    optimizer: torch.optim - the optimizer to be used
    device: str - the device to be used for training
    """
    model = model.to(device)
    for epoch in range(num_epochs):
        with tqdm(
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            position=0,
            leave=True,
        ) as pbar:
            for step, batch in enumerate(dataloader):
                images = batch[0]
                images = images.to(device)
                optimizer.zero_grad()

                t = torch.randint(0, const.T, (images.shape[0],), device=device).long()
                loss = get_loss(model, images, t)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())


def distributed_train():
    pass
