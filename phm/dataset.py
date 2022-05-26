

import glob
import logging
import os
import re

from pathlib import Path
from functools import lru_cache
from typing import Callable
from progress.bar import ChargingBar, Bar

from phm.data import MMEContainer, MMERecord
from phm.data.vtd import RGBDnT
from phm.io import supported_modality_loaders, save_mme, load_entity
from phm.io.mme import load_mme
from phm.io.vtd import load_RGBDnT, save_RGBDnT, save_dual_point_cloud, save_point_cloud
from phm.vtd import VTD_Alignment
from phm.utils import ftype_to_filext

class Dataset:
    def __init__(self) -> None:
        super().__init__()
        self.__index = 0

    def get(self, index : int):
        raise NotImplementedError()
        
    def reset(self):
        self.__index = 0

    def __getitem__(self, index : int):
        return self.get(index)
    
    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        return self
    
    def __next__(self):
        data = self.get(self.__index)
        self.__index += 1
        return data

class Dataset_LoadableFunc(Dataset):
    def __init__(self, 
        in_dir : str,
        file_type : str,
        load_func : Callable
    ) -> None:
        super().__init__()
        self.in_dir = in_dir
        self.file_type = file_type
        self._load_func = load_func
        self.__init()

    def __init(self):
        # List the files
        file_extension = ftype_to_filext(self.file_type)
        vfiles = glob.glob(os.path.join(self.in_dir, '*.' + file_extension))
        # Extract file ids and load entities
        self.data = []
        for f in vfiles:
            fname = os.path.basename(f)
            ptn = re.findall(r'\d{12}\d+', fname)
            if not ptn:
                logging.warning(f'{fname} does not follow the supported naming!')
                continue
            self.data.append((ptn[0], f))
        self.data.sort(key=lambda x : x[0])
        print(f'Found {len(self.data)} {self.file_type} items.')

    def __len__(self):
        return len(self.data)
    
    @lru_cache(maxsize=10)
    def get_by_fid(self, fid : str):
        index = 0
        try:
            index = [x[0] for x in self.data].index(fid)
            return self.get(index)
        except ValueError as ex:
            logging.exception(ex)
            raise ValueError(f'fid ({fid}) does not exist!')

    @lru_cache(maxsize=10)
    def get(self, index : int):
        if index >= len(self):
            raise StopIteration
        fid, file = self.data[index]
        return (fid, self._load_func(file, self.file_type))

class VTD_Dataset(Dataset_LoadableFunc):
    def __init__(self, in_dir: str) -> None:
        super().__init__(in_dir, 'mat', load_RGBDnT)

    @lru_cache(maxsize=10)
    def get(self, index : int):
        if index >= len(self):
            raise StopIteration
        fid, file = self.data[index]
        return (fid, self._load_func(file))

def create_mme_dataset(
    root_dir : str, 
    res_dir : str,
    file_type : str
):
    # Check the validity of root directory
    if root_dir is None or not os.path.isdir(root_dir):
        raise ValueError(f'{root_dir} is an invalid directory path!')
    # Find directories associated with the data types
    sub_folders = {}
    for dtype in supported_modality_loaders():
        dtype_dir = os.path.join(root_dir, str(dtype))
        if os.path.isdir(dtype_dir):
            sub_folders[dtype] = dtype_dir

    if not sub_folders:
        raise ValueError('No supported modalities has been found!')
    
    existing_types = tuple(sub_folders.keys())
    # Create the result directory
    file_extension = file_type
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    # List all visible images
    if not 'visible' in existing_types:
        raise ValueError('Visible modality does not found!')
    
    vfiles = glob.glob(os.path.join(sub_folders['visible'], '*.png'))
    vfiles.sort(key=os.path.getmtime)
    # Extract file ids
    file_ids = []
    for x in vfiles:
        fname = os.path.basename(x)
        ptn = re.findall(r'\d{12}\d+', fname)
        if not ptn:
            logging.warning(f'{fname} does not follow the supported naming!')
            continue
        file_ids.append(ptn[0])
    # Generate the files
    matched = 0

    with Bar('Creating MME Dataset', max=len(file_ids)) as bar:
        for ptn in file_ids:
            container = MMEContainer(cid=ptn)
            # Check if find all modalities
            modalities = {}
            for (dtype, dfolder) in sub_folders.items():
                fname = f'{str(dtype)}_{ptn}.png'
                full_path = os.path.join(dfolder, fname)
                if not os.path.isfile(full_path):
                    # logging.warning(f'{fname} does not exist!')
                    continue
                modalities[dtype] = full_path

            if len(modalities) == len(existing_types):
                matched += 1        
                # Add modalities to the container
                for (dtype, fpath) in modalities.items():
                    container.add_entity(MMERecord(
                        type=dtype,
                        file=fpath,
                        data=load_entity(dtype, fpath)
                    ))
                # Save the container
                save_mme(
                    os.path.join(res_dir, f'mme_{ptn}.{file_extension}'),
                    record=container,
                    file_type=file_type
                )
            bar.next()

    print(f'Total : {len(file_ids)}, Matched : {matched}')

def create_vtd_dataset(
    in_dir : str,
    target_dir : str,
    depth_param_file : str,
    in_type : str,
    homography_fid : str = None):
    
    if not os.path.isdir(in_dir):
        raise ValueError('Data directory does not exist!')
    
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    align = VTD_Alignment(
        target_dir=target_dir,
        depth_param_file=depth_param_file
    )
    dataset = Dataset_LoadableFunc(in_dir, in_type, load_mme)
    # Estimate Alignment Parameters
    if homography_fid is not None:
        align.estimate_alignment_params(dataset.get_by_fid(homography_fid))

    with Bar('Creating VTD Dataset', max=len(dataset)) as bar:
        for x in dataset:
            fid = x[0]
            data = x[1]
            res = align.compute(data)
            save_RGBDnT(os.path.join(target_dir, f'vtd_{fid}.mat'), res)
            bar.next()

def create_point_cloud_dataset(
    in_dir : str,
    target_dir : str,
    file_type : str
):
    if not os.path.isdir(in_dir):
        raise ValueError('RGBD&T directory does not exist!')
    
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    dataset = VTD_Dataset(in_dir)
    file_extension = ftype_to_filext(file_type)

    with Bar('Creating Point Cloud Dataset', max=len(dataset)) as bar:
        for x in dataset:
            fid = x[0]
            data = x[1]
            save_point_cloud(
                file=os.path.join(target_dir,f'pcs_{fid}.{file_extension}'),
                data=data,
                file_type=file_type
            )
            bar.next()

def create_dual_point_cloud_dataset(in_dir : str, target_dir : str):
    if not os.path.isdir(in_dir):
        raise ValueError('RGBD&T directory does not exist!')
    
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    dataset = VTD_Dataset(in_dir)

    with Bar('Processing', max=len(dataset)) as bar:
        for x in dataset:
            fid = x[0]
            data = x[1]
            save_dual_point_cloud(data, fid, target_dir)
            bar.next()