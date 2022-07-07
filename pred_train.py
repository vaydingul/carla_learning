import torch
from torch.utils.data import DataLoader

from expert_dataset import ExpertDataset, LearningType
from models.affordance_predictor import AffordancePredictor
import matplotlib.pyplot as plt


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    model.eval()
    test_loss = 0
    counter = 0
    with torch.no_grad():
        for batch in dataloader:
            image, command,  lane_dist, lane_angle, tl_dist, tl_state = batch
            lane_dist_pred, lane_angle_pred, tl_dist_pred, tl_state_pred = model(image, command)
            loss = model.loss(lane_dist_pred, lane_dist, lane_angle_pred, lane_angle, tl_dist_pred, tl_dist, tl_state_pred, tl_state)
            test_loss += loss.item()
            counter += image.shape[0] # batch size

    # Report average loss on the validation dataset
    return test_loss / counter


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    model.train()

    for batch in dataloader:
        model.optimizer.zero_grad()

        image, command,  lane_dist, lane_angle, tl_dist, tl_state = batch
        lane_dist_pred, lane_angle_pred, tl_dist_pred, tl_state_pred = model(image, command)
        loss = model.loss(lane_dist_pred, lane_dist, lane_angle_pred, lane_angle, tl_dist_pred, tl_dist, tl_state_pred, tl_state)
        loss.backward()
        model.optimizer.step()

    # Report the latest loss on that epoch
    return loss.item()


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    plt.figure()
    plt.plot(train_loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig("losses.png")
    plt.show()

def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = os.path.join("dataset", "train")
    val_root = os.path.join("dataset", "val")
    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root, LearningType.AFFORDANCE)
    val_dataset = ExpertDataset(val_root, LearningType.AFFORDANCE)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
