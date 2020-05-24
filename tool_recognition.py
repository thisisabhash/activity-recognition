import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data import DataLoader
import time
import copy
import datetime
import config
import dataset

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in [config.PHASE_TRAIN, config.PHASE_VALIDATION]:
            if phase == config.PHASE_TRAIN:
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for batch in iter(dataloaders[phase]):
                inputs = batch[config.DATASET_KEYS_IMAGE]
                labels = batch[config.DATASET_KEYS_TOOLS]

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == config.PHASE_TRAIN):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs)

                    loss = criterion(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == config.PHASE_TRAIN:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # compute running corrects
                for i in range(len(preds)):
                    prediction = preds[i]
                    label = labels[i]
                    corrects = 0.0
                    for j in range(len(prediction)):
                        if prediction[j] > 0.5 and label[j] == 1:
                            corrects += 1

                        elif prediction[j] < 0.5 and label[j] == 0:
                            corrects += 1
                    running_corrects += (corrects/7)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == config.PHASE_VALIDATION and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloaders, device, dataset_sizes):
    model.eval()
    running_corrects = 0.0

    with torch.no_grad():
        for batch in iter(dataloaders[config.PHASE_TEST]):
            inputs = batch[config.DATASET_KEYS_IMAGE]
            labels = batch[config.DATASET_KEYS_TOOLS]

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.sigmoid(outputs)

            # compute running corrects
            for i in range(len(preds)):
                prediction = preds[i]
                label = labels[i]
                corrects = 0.0
                for j in range(len(prediction)):
                    if prediction[j] > 0.5 and label[j] == 1:
                        corrects += 1

                    elif prediction[j] < 0.5 and label[j] == 0:
                        corrects += 1
                running_corrects += (corrects / 7)

    test_acc = running_corrects / dataset_sizes[config.PHASE_TEST]
    print('Test accuracy: {}'.format(test_acc))


def main():
    print('Tool Recognition start: ' + str(datetime.datetime.now()))
    
    image_datasets = {x: dataset.tool_data(config.DATA_DIR, x, config.TRANSFORM)
                      for x in config.PHASES}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=8)
                   for x in config.PHASES}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in config.PHASES}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # change the code to load different models like resnet-18/34/50/100,
    # alexnet etc. from torchvision
    model_conv = torchvision.models.resnet152(pretrained=True)
    
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 7)
    
    criterion = nn.BCELoss()
    
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    
    print('Training model')
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                             exp_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=1)
    
    # Save model
    print('Saving model...')
    torch.save(model_conv.state_dict(), config.TOOL_MODEL_PATH)
    
    print('Testing model.')
    test_model(model_conv, dataloaders, device, dataset_sizes)
    
    print('Tool Recognition end: ' + str(datetime.datetime.now()))

if __name__ == '__main__':
    main()