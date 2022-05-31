
import os

from typing import List
from progress.bar import Bar

from phm.data import RGBDnT
from phm.io import load_RGBDnT

class Pipeline:
    def __init__(self, 
        root_dir : str, 
        filenames : List[str]
    ):
        self.root_dir = root_dir
        # Check file availability
        if len(filenames) == 0:
            raise ValueError('No RGBD&T file exist!')
        # Check Root Directory Availability
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f'{self.root_dir} does not exist!')
        # Initialize the file list
        self.files = tuple(filter(lambda x : os.path.isfile(os.path.join(self.root_dir, x)), filenames))

    @property
    def file_count(self):
        return len(self.files)

    def load_data_frames(self):
        # Load Data
        frames = list()
        with Bar('Loading Data ', max=self.file_count) as bar:
            for x in self.files:
                f = os.path.join(self.root_dir, x)
                frame = load_RGBDnT(f)
                frames.append(frame)
                bar.next()

        return frames

    def execute(self):
        raise NotImplementedError('Execute method is not implemented yet!')