import os
from skimage import io
from torch.utils.data.dataset import Dataset
import config
import torch

# Custom Dataset classes used for different operations

# used in feature extraction
class feature_extract_data(Dataset):

    def __init__(self, directory, phase, transform=None):
        self.directory = directory
        self.phase = phase
        self.images, self.video_numbers = self.get_images(directory, phase)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = io.imread(img_name)

        number = self.video_numbers[idx]

        if self.transform:
            image = self.transform(image)

        sample = {config.DATASET_KEYS_IMAGE: image, config.DATASET_KEYS_NUMBER: number}
        return sample

    def __len__(self):
        return len(self.images)

    def get_images(self, directory, phase):
        if phase == config.PHASE_TEST:
            lower_bound = config.TEST_SET_LOWER_BOUND
            upper_bound = config.TEST_SET_UPPER_BOUND

        else:
            lower_bound = config.TRAIN_SET_LOWER_BOUND
            upper_bound = config.TRAIN_SET_UPPER_BOUND

        image_names = list()
        video_numbers = list()

        for i in range(lower_bound, upper_bound):
            image_num = 0
            while (1):
                image_path = ('{}/video{}/image{}.jpg').format(directory, str(i).zfill(2), str(image_num))
                if os.path.isfile(image_path):
                    image_names.append(image_path)
                    video_numbers.append(str(i).zfill(2))
                    image_num += 25
                else:
                    break
        return image_names, video_numbers

# used in tool recognition
class tool_data(Dataset):
    def __init__(self, directory,phase, transform=None):
        self.directory = directory
        self.phase = phase
        self.images = self.get_images(directory, phase)
        self.tool_annotations = self.get_tool_annotations(directory, phase)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = io.imread(img_name)
        tool_list = self.tool_annotations[idx]
        tools = list()
        for tool in tool_list:
            tools.append(int(tool))

        tools = torch.FloatTensor(tools)
        if self.transform:
            image = self.transform(image)

        sample = {config.DATASET_KEYS_IMAGE: image, config.DATASET_KEYS_TOOLS: tools}
        return sample

    def get_images(self, directory, phase):
        if phase == config.PHASE_TEST:
            lower_bound = config.TEST_SET_LOWER_BOUND
            upper_bound = config.TEST_SET_UPPER_BOUND
        else:
            lower_bound = config.TRAIN_SET_LOWER_BOUND
            upper_bound = config.TRAIN_SET_UPPER_BOUND

        image_names = list()

        for i in range(lower_bound, upper_bound):
            image_num = 0
            while (1):
                image_path = ('{}/video{}/image{}.jpg').format(directory, str(i).zfill(2), str(image_num))
                if os.path.isfile(image_path):
                    image_names.append(image_path)
                    image_num += 25
                else:
                    break
            # pops last element from list
            # As the last frame isn't annotated in the dataset
            image_names.pop()
        return image_names

    def get_tool_annotations(self, directory, phase):
        if phase == config.PHASE_TEST:
            lower_bound = config.TEST_SET_LOWER_BOUND
            upper_bound = config.TEST_SET_UPPER_BOUND
        else:
            lower_bound = config.TRAIN_SET_LOWER_BOUND
            upper_bound = config.TRAIN_SET_UPPER_BOUND

        tool_annotations = list()
        for i in range(lower_bound, upper_bound):
            tool_path = '{}/video{}-tool.txt'.format(directory, str(i).zfill(2))
            if os.path.isfile(tool_path):
                with open(tool_path, 'r') as file:
                    next(file)
                    image_num = 0
                    for line in file:
                        line = line.strip('\n').split('\t')
                        line.remove(str(image_num))
                        image_num += 25
                        tool_annotations.append(line)

        return tool_annotations


# used in phase recognition
class phase_data(Dataset):

    def __init__(self, directory, phase, transform=None):
        self.directory = directory
        self.sequence_length = 200
        self.phase = phase
        self.images = self.get_images(directory, phase)
        self.phase_annotations = self.get_phase_annotations(directory, phase)
        if len(self.images) != len(self.phase_annotations):
            print("{} Length of images not same as length of annotations".format(self.phase))
        self.transform = transform

    def __getitem__(self, idx):
        img_sequence = self.images[idx]
        images = torch.randn(len(img_sequence), 1, 2048)

        for i in range(len(img_sequence)):
            line = img_sequence[i]
            line = line.split(' ')
            for j in range(len(line)):
                images[i][0][j] = float(line[j])

        images = torch.FloatTensor(images)
        phases = torch.FloatTensor(self.phase_annotations[idx])

        sample = {config.DATASET_KEYS_IMAGE: images, config.DATASET_KEYS_PHASES: phases}
        return sample

    def __len__(self):
        return len(self.images)

    def get_images(self, directory, phase):
        if phase == config.PHASE_TEST:
            lower_bound = config.TEST_SET_LOWER_BOUND
            upper_bound = config.TEST_SET_UPPER_BOUND
        else:
            lower_bound = config.TRAIN_SET_LOWER_BOUND
            upper_bound = config.TRAIN_SET_UPPER_BOUND

        image_features = list()

        for i in range(lower_bound, upper_bound):
            video_path = '{}/video_features{}.txt'.format(directory, str(i).zfill(2))
            if os.path.isfile(video_path):
                with open(video_path, 'r') as file:
                    for line in file:
                        image_features.append(line.strip('\n'))

        image_sequences = list()
        
        length = int(len(image_features)/self.sequence_length)
        print('image_features length: '+ str(len(image_features)))
        print(length)
        for i in range(length):
            sequence = list()
            start = self.sequence_length*i
            for j in range(start, start+self.sequence_length):
                sequence.append(image_features[j])
            image_sequences.append(sequence)

        return image_sequences

    def get_phase_annotations(self, directory, phase):
        if phase == config.PHASE_TEST:
            lower_bound = config.TEST_SET_LOWER_BOUND
            upper_bound = config.TEST_SET_UPPER_BOUND
        else:
            lower_bound = config.TRAIN_SET_LOWER_BOUND
            upper_bound = config.TRAIN_SET_UPPER_BOUND

        phase_annotations = list()
        for i in range(lower_bound, upper_bound):
            phase_path = '{}/video{}-phase.txt'.format(config.DATA_DIR, str(i).zfill(2))
            if os.path.isfile(phase_path):
                with open(phase_path, 'r') as file:
                    next(file)
                    image_num = 0
                    for line in file:
                        if image_num == 0 or image_num % 25 == 0:
                            line = line.strip('\n').split('\t')
                            if str(image_num) not in line:
                                continue
                            line.remove(str(image_num))
                            line = config.PHASE_TO_INDEX[line[0]]
                            phase_annotations.append(line)
                        image_num += 625
        
        phase_sequences = list()
        length = int(len(phase_annotations)/self.sequence_length)
        print('Phase annotation length: '+ str(len(phase_annotations)))
        print(length)
        for i in range(length):
            sequence = list()
            start = self.sequence_length*i
            for j in range(start, start+self.sequence_length):
                sequence.append(phase_annotations[j])
            phase_sequences.append(sequence)
        return phase_sequences

