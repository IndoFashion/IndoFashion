"""
    Script File to train for Cloth Classification task
"""
import os
import torch
import torch.utils.data
from tqdm import tqdm
import torch.optim as optim
import utils.config as config
from model_archs.img_models import ResNetFeaturesFlatten
from utils import tflogger
from utils.common_utils import image_transform, image_transform_jitter_flip, get_accuracy
from utils.dataset import EthnicFinderDataset
import numpy as np

model_name = "iew_r50_jitter_flip"

# Dataloaders
train_set = EthnicFinderDataset(metadata_file=config.train_file, mode='train', transform=image_transform_jitter_flip)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
val_set = EthnicFinderDataset(metadata_file=config.val_file, mode='val', transform=image_transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)
test_set = EthnicFinderDataset(metadata_file=config.test_file, mode='test', transform=image_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=True, num_workers=2)

# Model
cloth_model = ResNetFeaturesFlatten(model_key="resnet50").to(config.device)

# Optimizer
optimizer = optim.Adam(cloth_model.parameters(), lr=config.lr)
print("Total Params", sum(p.numel() for p in cloth_model.parameters() if p.requires_grad))

# Logger
logger = tflogger.Logger(model_name=model_name, data_name='ours',
                         log_path=os.path.join(config.BASE_DIR, 'tf_logs', model_name))


def train_epoch(epoch):
    """
        Runs one training epoch
        Args:
            epoch (int): Epoch number
    """
    train_loss = 0.
    total = 0.
    correct = 0.
    cloth_model.train()
    # Training loop
    for batch_idx, (image, label) in enumerate(tqdm(train_loader)):
        image = image.to(config.device)
        label = label.to(config.device)

        batch = image.shape[0]
        with torch.set_grad_enabled(True):
            y_pred = cloth_model(image)
            loss = config.ce_criterion(y_pred, label)
            loss.backward()
            train_loss += float(loss.item())
            optimizer.step()
            optimizer.zero_grad()  # clear gradients for this training step
            correct += get_accuracy(y_pred, label)
            total += batch
            torch.cuda.empty_cache()
            del image, label, y_pred

    # Calculate loss and accuracy for current epoch
    logger.log(mode="train", scalar_value=train_loss / len(train_loader), epoch=epoch, scalar_name='loss')
    logger.log(mode="train", scalar_value=correct / total, epoch=epoch, scalar_name='accuracy')

    print(' Train Epoch: {} Loss: {:.4f} Acc: {:.2f} '.format(epoch, train_loss / len(train_loader), correct / total))


def eval_epoch(epoch):
    """
        Runs one evaluation epoch
        Args:
            epoch (int): Epoch number
    """
    cloth_model.eval()
    val_loss = 0.
    total = 0.
    correct = 0.
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(val_loader, desc='')):
            image = image.to(config.device)
            label = label.to(config.device)
            batch = image.shape[0]
            y_pred = cloth_model(image)
            loss = config.ce_criterion(y_pred, label)
            val_loss += float(loss.item())
            correct += get_accuracy(y_pred, label)
            total += batch
            torch.cuda.empty_cache()
            del image, label, y_pred

        logger.log(mode="val", scalar_value=val_loss / len(val_loader), epoch=epoch, scalar_name='loss')
        logger.log(mode="val", scalar_value=correct / total, epoch=epoch, scalar_name='accuracy')

        print(' Val Epoch: {} Avg loss: {:.4f} Acc: {:.2f}'.format(epoch, val_loss / len(val_loader), correct / total))
    return val_loss


def train_model():
    """
        Trains and evaluates the classification model
    """
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(config.BASE_DIR + 'models/' + model_name + '.pt')
        cloth_model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        cloth_model.eval()
        best_loss = eval_classifier_full()
    except:
        best_loss = np.Inf
    early_stop = False
    counter = 0
    for epoch in range(1, config.epochs + 1):
        # Training epoch
        train_epoch(epoch)
        # Validation epoch
        avg_test_loss = eval_epoch(epoch)
        if avg_test_loss <= best_loss:
            counter = 0
            best_loss = avg_test_loss
            torch.save(cloth_model.state_dict(), 'models/' + model_name + '.pt')
            print("Best model saved/updated..")
            torch.cuda.empty_cache()
        else:
            counter += 1
            if counter >= config.patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            break


def eval_classifier_full():
    """
        Evaluates validation loss for computing previous best loss when model weights are loaded from memory
    """
    val_loss = 0.
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(val_loader, desc='')):
            image = image.to(config.device)
            label = label.to(config.device)
            y_pred = cloth_model(image)
            loss = config.ce_criterion(y_pred, label)
            val_loss += loss.item()
            torch.cuda.empty_cache()
            del image, label, y_pred
        print(' Val Avg loss: {:.4f}'.format(val_loss / len(val_loader)))
    return val_loss


def test_classifier():
    """
        Test Classification Accuracy on test set after training
    """
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(config.BASE_DIR + 'models/' + model_name + '.pt')
        cloth_model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        cloth_model.eval()
    except:
        print("Model Not Found")
        exit()
    total = 0.
    correct = 0.
    labels = torch.LongTensor([])
    y_preds = torch.LongTensor([])

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(test_loader, desc='')):
            image = image.to(config.device)
            label = label.to(config.device)
            batch = image.shape[0]
            y_pred = cloth_model(image)
            _, y_pred_cls = torch.max(y_pred, 1)
            labels = torch.cat([labels, label.cpu()])
            y_preds = torch.cat([y_preds, y_pred_cls.cpu()])
            correct += get_accuracy(y_pred, label)
            total += batch
            torch.cuda.empty_cache()
        print('Test Acc: {:.4f}'.format(correct / total))


if __name__ == '__main__':
    # train_model()
    test_classifier()
