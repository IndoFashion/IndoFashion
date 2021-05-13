from __future__ import print_function, division
import cv2
import random
from torch.utils.data import Dataset
from utils.common_utils import read_json_data, CLOTH_CATEGORIES
from utils.config import DATA_DIR


class EthnicFinderDataset(Dataset):
    """Custom dataset class for Cloth Classsification"""

    def __init__(self, metadata_file, mode, transform=None):
        """

        Args:
            metadata_file (string): Path to the json file with annotations.
            mode (string): train, test, val
            transform (callable): Transform to be applied on a sample.

        Returns:
            None
        """
        self.metadata_list = read_json_data(metadata_file)
        self.mode = mode
        self.transform = transform

    def __getitem__(self, idx):
        """
            Returns sample corresponding to the index `idx`
        """
        metadata = self.metadata_list[idx]
        img_path = DATA_DIR + metadata["image_path"]
        class_label = metadata["class_label"]
        image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 256))
        label = CLOTH_CATEGORIES[class_label]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """
            Returns length of the dataset
        """
        return len(self.metadata_list)
