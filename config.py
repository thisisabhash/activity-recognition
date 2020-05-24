# Configuration file for project wide constants and common code
from torchvision import transforms


DATA_DIR = '/Users/abhash/Desktop/CS766 Computer Vision/Project/cholec80/Data'
TOOL_MODEL_PATH = '/Users/abhash/Desktop/CS766 Computer Vision/Project/cholec80/resnet152.pt'
PHASE_MODEL_PATH = '/Users/abhash/Desktop/CS766 Computer Vision/Project/cholec80/phase_tagger.pt'
NUM_VIDEOS = 80 #cholec80 has total 80 videos

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Video indexes for training and test sets
TRAIN_SET_LOWER_BOUND = 1
TRAIN_SET_UPPER_BOUND = 40

TEST_SET_LOWER_BOUND = 40
TEST_SET_UPPER_BOUND = 79

PHASE_TEST = 'Test'
PHASE_TRAIN = 'Train'
PHASE_VALIDATION = 'Validation'
PHASES = [PHASE_TEST, PHASE_TRAIN, PHASE_VALIDATION]
TOOL_CLASSES = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'SpecimenBag']

DATASET_KEYS_IMAGE = 'images'
DATASET_KEYS_TOOLS = 'tools'
DATASET_KEYS_NUMBER = 'number'
DATASET_KEYS_PHASES = 'phases'

PHASE_TO_INDEX = {'TrocarPlacement': 0, 
                  'Preparation': 1, 
                  'CalotTriangleDissection': 2, 
                  'ClippingCutting': 3,
                  'GallbladderDissection': 4, 
                  'GallbladderPackaging': 5, 
                  'CleaningCoagulation': 6,
                  'GallbladderRetraction': 7}
