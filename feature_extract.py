import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import config
import dataset
import datetime

def extract_features(model, dataloaders, device):

    with torch.no_grad():
        for phase in config.PHASES[:-1]:
            video_number = '0'
            output_string = ''
            for batch in iter(dataloaders[phase]):
                inputs = batch[config.DATASET_KEYS_IMAGE]
                number = batch[config.DATASET_KEYS_NUMBER]

                inputs = inputs.to(device)
                
                if number != video_number:
                    if video_number != '0':
                        file = open('{}/video_features{}.txt'.format(config.DATA_DIR, video_number[0]), 'w')
                        file.write(output_string)
                        file.close()
                        output_string = output_string.split(' ')
                    output_string = ''
                    video_number = number

                output = model(inputs)
                for i in range(2048):
                    output_value = output[0][i][0][0]
                    output_string += '%.4f' % (output_value.item())
                    if i == 2047:
                        output_string += '\n'
                    else:
                        output_string += ' '
            file = open('{}/video_features{}.txt'.format(config.DATA_DIR, video_number[0]), 'w')
            file.write(output_string)
            file.close()
            output_string = output_string.split(' ')

def main():
    print('Feature extraction start: ' + str(datetime.datetime.now()))
    
    image_datasets = {x: dataset.feature_extract_data(config.DATA_DIR, x, config.TRANSFORM) for x in config.PHASES}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=False, num_workers=0)
                   for x in config.PHASES}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    model_conv = torchvision.models.resnet152()
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 7)
    model_conv = model_conv.to(device)
    model_conv.load_state_dict(torch.load(config.TOOL_MODEL_PATH))
    model_conv.eval()
    model_conv = torch.nn.Sequential(*(list(model_conv.children())[:-1]))
    extract_features(model_conv, dataloaders, device)
    
    print('Feature extraction end: ' + str(datetime.datetime.now()))

if __name__ == '__main__':
    main()