from pathlib import Path
import torch
from torch.utils.data import DataLoader
import os
from expert_dataset import ExpertDataset, LearningType
from models.cilrs import CILRS
import matplotlib.pyplot as plt
import yaml
import tqdm

def get_config(file_path):
    """Get the configuration of the model"""
    with open(file_path, "r") as f:
        config = yaml.load(f)
    return config
    
def validate(model, dataloader):
    """Validate CILRS model performance on the validation dataset"""
    model.eval()
    model.to('cuda:0')

    test_loss = 0
    counter = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, colour="blue"):
            image, command, speed, steer, throttle, brake = batch
            speed_pred, action_pred = model(image.to('cuda:0'), speed.to('cuda:0'), command.to('cuda:0'))
            loss = model.loss_criterion(speed_pred, speed.to('cuda:0'), action_pred, torch.cat((steer.to('cuda:0'), throttle.to('cuda:0') - brake.to('cuda:0')), dim=1))
            test_loss += loss.item()
            counter += image.shape[0] # batch size

    # Report average loss on the validation dataset
    return test_loss / counter


def train(model, dataloader):
    """Train CILRS model on the training dataset for one epoch"""
    model.train()
    model.to('cuda:0')
    train_loss = 0
    counter = 0
    for batch in tqdm.tqdm(dataloader, colour="green"):
        model.optimizer.zero_grad()

        image, command, speed, steer, throttle, brake = batch
        speed_pred, action_pred = model(image.to('cuda:0'), speed.to('cuda:0'), command.to('cuda:0'))
        loss = model.loss_criterion(speed_pred, speed.to('cuda:0'), action_pred, torch.cat((steer.to('cuda:0'), throttle.to('cuda:0') - brake.to('cuda:0')), dim=1))
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        counter += image.shape[0] # batch size
    # Report the latest loss on that epoch
    return train_loss / counter



def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    plt.figure()
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("losses_cilrs.png")
    plt.show()

    
def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = os.path.join("dataset/train", "expert")
    val_root = os.path.join("dataset/val", "expert")

    model = CILRS(get_config(os.path.join(Path("configs"), "cilrs_network.yaml")))

    train_dataset = ExpertDataset(train_root, LearningType.IMITATION)
    val_dataset = ExpertDataset(val_root, LearningType.IMITATION)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 128
    save_path = "cilrs_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in tqdm.tqdm(range(num_epochs), colour="red"):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
        print(train_losses[-1], val_losses[-1])
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
