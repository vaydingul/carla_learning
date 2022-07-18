from pathlib import Path
import torch
from torch.utils.data import DataLoader
import os
from expert_dataset import ExpertDataset, LearningType
from models.cilrs import CILRS
import matplotlib.pyplot as plt
import yaml
import tqdm
import wandb
from datetime import datetime
import argparse

def get_config(file_path):
    """Get the configuration of the model"""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config
    
def validate(model, dataloader, epoch, run):
    """Validate CILRS model performance on the validation dataset"""
    model.eval()
    model.to('cuda:0')

    test_loss = 0
    test_action_loss = 0
    test_speed_loss = 0
    counter = 0
    with torch.no_grad():
        for batch in dataloader:
            image, command, speed, steer, throttle, brake = batch
            speed_pred, action_pred = model(image.to('cuda:0'), speed.to('cuda:0'), command.to('cuda:0'))
            loss, speed_loss, action_loss = model.loss_criterion(speed_pred, speed.to('cuda:0'), action_pred, torch.cat((steer.to('cuda:0'), throttle.to('cuda:0') - brake.to('cuda:0')), dim=1))
            test_loss += loss.item()
            test_action_loss += action_loss.item()
            test_speed_loss += speed_loss.item()
            step = epoch * len(dataloader.dataset) + counter * dataloader.batch_size + image.shape[0]

            counter += 1#image.shape[0] # batch size
        
    # Report average loss on the validation dataset
    average_loss = test_loss / counter
    average_action_loss = test_action_loss / counter
    average_speed_loss = test_speed_loss / counter

    run.log({"val/step": step, "val/loss": average_loss, "val/speed_loss": average_speed_loss, "val/action_loss": average_action_loss}, step=step)

    return average_loss


def train(model, dataloader, epoch, run):
    """Train CILRS model on the training dataset for one epoch"""
    model.train()
    model.to('cuda:0')
    train_loss = 0
    counter = 0
    for batch in dataloader:
        model.optimizer.zero_grad()

        image, command, speed, steer, throttle, brake = batch
        speed_pred, action_pred = model(image.to('cuda:0'), speed.to('cuda:0'), command.to('cuda:0'))
        loss, speed_loss, action_loss = model.loss_criterion(speed_pred, speed.to('cuda:0'), action_pred, torch.cat((steer.to('cuda:0'), throttle.to('cuda:0') - brake.to('cuda:0')), dim=1))
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        step = epoch * len(dataloader.dataset) + counter * dataloader.batch_size + image.shape[0]
        run.log({"train/step": step, "train/loss": loss.item(), "train/speed_loss": speed_loss.item(), "train/action_loss": action_loss.item()}, step=step)
        counter += 1#image.shape[0] # batch size
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

    
def main(config_path, train_path, val_path):

    

    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = Path(train_path)
    val_root = Path(val_path)

    model_config = get_config(os.path.join(Path("configs"), config_path))
    model = CILRS(model_config)


    train_dataset = ExpertDataset(train_root, LearningType.IMITATION)
    val_dataset = ExpertDataset(val_root, LearningType.IMITATION)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = model_config["num_epochs"]
    batch_size = model_config["batch_size"]
    
    os.makedirs("ckpts", exist_ok=True)
    
    
    run = wandb.init(project="carla_learning", group = "cilrs", name="cilrs_train__", config = model_config)
    run.define_metric("train/step")
    run.define_metric("val/step")
    run.define_metric(name = "train/*", step_metric = "train/step")
    run.define_metric(name = "val/*", step_metric = "val/step")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    run.alert("Training started", "Training started")
    for i in range(num_epochs):

        train_losses.append(train(model, train_loader, i, run))
        val_losses.append(validate(model, val_loader, i, run))
        
        if ((i+1) % 5) == 0:

            run.alert("Epoch-wise Info", "Epoch {}/{}".format(i + 1, num_epochs))

            # Save path is the save path from config + date time in string format
            datestr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
            save_path = model_config["save_path"] + datestr + ".ckpt"
            save_path_ = os.path.join(Path("ckpts"), str(i+1) + "-" + save_path)

            torch.save(model, save_path_)
            
    plot_losses(train_losses, val_losses)
    run.save(save_path_)
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cilrs_network.yaml", help="Path to config file")
    parser.add_argument("--train_path", default="dataset_expert", help="Path to training dataset")
    parser.add_argument("--val_path", default="dataset_expert", help="Path to validation dataset")

    args = parser.parse_args()
    main(args.config, args.train_path, args.val_path)
