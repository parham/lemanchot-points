
import re
import os
import glob
import logging
import numpy as np

from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

modal_to_image = lambda img : (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255.0).astype(np.uint8)
gray_to_rgb = lambda img : np.stack((img, img, img), axis=2)

__modality_loaders = {}

def modality_loader(name : Union[str, List[str]]):
    def __embed_func(func):
        global __modality_loaders
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __modality_loaders[n] = func
    return __embed_func

def supported_modality_loaders() -> Tuple:
    return tuple(__modality_loaders.keys())

@lru_cache(maxsize=4)
def load_entity(file_type : str, file : str):
    if not os.path.isfile(file):
        raise ValueError(f'{file} is invalid!')

    if not file_type in supported_modality_loaders():
        raise ValueError(f'{file_type} loader does not exist!')

    return __modality_loaders[file_type](file, file_type)
@dataclass
class MMERecord:
    data : Any
    file : str
    type : str
class MMEContainer(object):
    def __init__(self, 
        cid : str = '',
        *entities : MMERecord,
        metadata : Dict = None
    ) -> None:
        self.container_id = cid
        self._entities = {}
        self._metadata = {}
        # Add the metadata
        if metadata is not None:
            self.set_metadatas(metadata)
        # Add entities
        if entities is not None:
            for e in entities:
                self.add_entity(e.type,)
    
    def add_entity_detailed(self, type : str, file : str, data):
        self.add_entity_record(MMERecord(data, file, type))

    def add_entity(self, type : str, file : str, overwrite : bool = False):
        data = load_entity(type, file)
        entity = MMERecord(
            data = data,
            file = file,
            type = type
        )
        self.add_entity_record(entity, overwrite=overwrite)

    def add_entity_record(self, entity : MMERecord, overwrite : bool = False):
        if entity.type in self._entities and not overwrite:
            raise ValueError(f'Type {type} already exist!')
        self._entities[entity.type] = entity

    def get_entity(self, type : str) -> MMERecord:
        if not type in self._entities:
            raise ValueError(f'Type {type} does not exist!')
        return self._entities[type]
    
    def get_entities(self):
        return tuple(self._entities.values())

    @property
    def modality_names(self):
        return tuple(self._entities.keys()) 

    @property
    def modalities(self):
        return tuple(self._entities.values())

    def __getitem__(self, type : str) -> Any:
        return self.get_entity(type)
    
    def __setitem__(self, type : str, entity : MMERecord):
        self.add_entity(type, entity, overwrite=False)

    def set_metadata(self, key : str, value : Union[int, float, str, bool]):
        if not isinstance(value, (bool, int, float, str)):
            raise ValueError(f'The metadata (key)\'s type is not supported!')
        self._metadata[key] = value
    
    def set_metadatas(self, metadata : Dict):
        self._metadata = {**self._metadata, **metadata}

    def get_metadata(self) -> Dict:
        return self._metadata

__mme_exporters = {}

def mme_exporter(name : Union[str, List[str]]):
    def __embed_func(func):
        global __mme_exporters
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __mme_exporters[n] = func
    return __embed_func

def supported_mme_exporters() -> Tuple:
    return tuple(__mme_exporters.keys())

def save_mme(file : str, record : MMEContainer, file_type : str):
    if not file_type in supported_mme_exporters():
        raise ValueError(f'{file_type} loader does not exist!')
    return __mme_exporters[file_type](file, record, file_type)

__mme_loaders = {}

def mme_loader(name : Union[str, List[str]]):
    def __embed_func(func):
        global __mme_loaders
        hname = name if isinstance(name, list) else [name]
        for n in hname:
            __mme_loaders[n] = func
    return __embed_func

def supported_mme_loaders() -> Tuple:
    return tuple(__mme_loaders.keys())

def load_mme(file : str, file_type : str):
    if not file_type in supported_mme_loaders():
        raise ValueError(f'{file_type} loader does not exist!')
    return __mme_loaders[file_type](file, file_type)

def create_mme_dataset(
    root_dir : str, 
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
    res_dir = os.path.join(root_dir, file_type)
    file_extension = file_type
    Path(res_dir).mkdir(parents=True, exist_ok=True)
    # List all visible images
    if not 'visible' in existing_types:
        raise ValueError('Visible modality does not found!')
    vfiles = glob.glob(os.path.join(sub_folders['visible'], '*.png'))
    vfiles.sort(key=os.path.getmtime)
    # Extract file ids
    # file_ids = [re.findall('\d{12}\d+', os.path.basename(x))[0] for x in vfiles]
    file_ids = []
    for x in vfiles:
        fname = os.path.basename(x)
        ptn = re.findall('\d{12}\d+', fname)
        if not ptn:
            logging.warning(f'{fname} does not follow the supported naming!')
            continue
        file_ids.append(ptn[0])
    # Generate the files
    matched = 0
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
                container.add_entity(dtype, fpath)
            # Save the container
            save_mme(
                os.path.join(res_dir, f'mme_{ptn}.{file_extension}'),
                record=container,
                file_type=file_type
            )

    print(f'Total : {len(file_ids)}, Matched : {matched}')