import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import copy
import config
import datetime
import dataset
import phase_model

def train_model(model, criterion, optimiser, dataloaders, dataset_sizes, device, num_epochs=4):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        model.train()

        running_loss = 0.0
        running_corrects = 0.0

        for batch in iter(dataloaders[config.PHASE_TRAIN]):

            inputs = batch[config.DATASET_KEYS_IMAGE]
            phases = batch[config.DATASET_KEYS_PHASES]

            inputs = inputs.to(device)
            phases = phases.to(device)

            # zero the parameter gradients
            optimiser.zero_grad()

            model.hidden = model.init_hidden()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                output = model(inputs[0])

                output = output.to(device)
                loss = criterion(output, (phases.long())[0])

                # backward + optimise only if in training phase
                loss.backward()
                optimiser.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            # compute prediction
            sequence_corrects = 0.0
            for i in range(len(output)):
                predictions = output[i]
                correct_phase = phases[0][i]

                largest_prediction = 0
                largest_prediction_index = 0
                for j in range(len(predictions)):
                    if predictions[j] > largest_prediction:
                        largest_prediction = predictions[j]
                        largest_prediction_index = j
                if largest_prediction_index == correct_phase:
                    sequence_corrects += 1

            sequence_corrects = sequence_corrects / len(output)
            running_corrects += sequence_corrects
        
        epoch_loss = running_loss / dataset_sizes[config.PHASE_TRAIN]
        epoch_acc = running_corrects / dataset_sizes[config.PHASE_TRAIN]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(config.PHASE_TRAIN, epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_model(model, dataloaders, dataset_sizes, device,):
    model.eval()

    running_corrects = 0.0

    with torch.no_grad():
        for batch in iter(dataloaders[config.PHASE_TEST]):
            inputs = batch[config.DATASET_KEYS_IMAGE]
            phases = batch[config.DATASET_KEYS_PHASES]

            inputs = inputs.to(device)
            phases = phases.to(device)

            model.hidden = model.init_hidden()

            output = model(inputs[0])

            # compute prediction
            sequence_corrects = 0.0
            for i in range(len(output)):
                predictions = output[i]
                correct_phase = phases[0][i]

                largest_prediction = 0
                largest_prediction_index = 0
                for j in range(len(predictions)):
                    if predictions[j] > largest_prediction:
                        largest_prediction = predictions[j]
                        largest_prediction_index = j
                if largest_prediction_index == correct_phase:
                    sequence_corrects += 1

            sequence_corrects = sequence_corrects / len(output)
            running_corrects += sequence_corrects

    test_acc = running_corrects / dataset_sizes['Test']
    print('Test accuracy: {}'.format(test_acc))


def main():
    print('Phase recognition start: ' + str(datetime.datetime.now()))
    
    image_datasets = {x: dataset.phase_data(config.DATA_DIR, x)
                      for x in config.PHASES[:-1]}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=0) 
                    for x in config.PHASES[:-1]}
    dataset_sizes = {x: len(image_datasets[x]) for x in config.PHASES[:-1]}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    softmax = nn.Softmax(2)
    
    # use different models to change architectures
    model = phase_model.RNN(2048, 64, 8, softmax, device)
    
    model = model.to(device)
    optimiser = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    model = train_model(model, criterion, optimiser, dataloaders, dataset_sizes, device, num_epochs=50)
    
    print('Saving model...')
    torch.save(model.state_dict(), config.PHASE_MODEL_PATH)
    print('Saved!')

    print('Testing')
    test_model(model, dataloaders, dataset_sizes, device)
    
    print('Phase recognition end: ' + str(datetime.datetime.now()))

if __name__ == '__main__':
    main()
