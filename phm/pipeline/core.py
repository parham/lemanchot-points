
import os

from typing import List
from progress.bar import Bar

from phm.data import RGBDnT
from phm.io import load_RGBDnT

class Pipeline:
    def __init__(self, file_list : List[str]):
        # Check file availability
        if len(self.file_list):
            raise ValueError('No RGBD&T file exist!')
            return
        # Initialize the file list
        self._flist = list()
        for f in file_list:
            if os.path.isfile(file_list):
                continue
            self._flist.append(f)

    def file_count(self):
        return len(self._flist)

    @property
    def file_list(self):
        return self._flist

    def execute(self):
        # 1. Load Data
        frames = self._load_data()
        # 2. Preprocessing RGBD&T data
        
    def _preprocessing(self, data):
        
    
    def _load_data(self):
        # Load Data
        frames = list()
        with Bar('Loading Data ', max=self.file_count) as bar:
            for f in self.file_list:
                frame = load_RGBDnT(f)
                data = frame.data
                frames.append(data)
                bar.next()

        return frames