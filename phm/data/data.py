
import glob
import json
import logging
from multiprocessing.sharedctypes import Value
import os
from pathlib import Path
import re
import zipfile

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Dict, Union
from zipfile import ZipFile
from scipy.io import savemat

from phm.data.core import MMEntityType, load_entity

@dataclass
class MMERecord:
    data : Any
    file : str
    type : MMEntityType

class MMEContainer(object):
    def __init__(self, 
        *entities : MMERecord,
        metadata : Dict = None
    ) -> None:
        self._entities = {}
        self._metadata = {}
        # Add the metadata
        if metadata is not None:
            self.set_metadatas(metadata)
        # Add entities
        if entities is not None:
            for e in entities:
                self.add_entity(e.type,)
    
    def add_entity(self, type : MMEntityType, file : str, overwrite : bool = False):
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

    def get_entity(self, type : MMEntityType) -> MMERecord:
        if not type in self._entities:
            raise ValueError(f'Type {type} does not exist!')
        return self._entities[type]
    
    def get_entities(self):
        return self._entities.values()

    def __getitem__(self, type : MMEntityType) -> Any:
        return self.get_entity(type)
    
    def __setitem__(self, type : MMEntityType, entity : MMERecord):
        self.add_entity(type, entity, overwrite=False)

    def set_metadata(self, key : str, value : Union[int, float, str, bool]):
        if not isinstance(value, (bool, int, float, str)):
            raise ValueError(f'The metadata (key)\'s type is not supported!')
        self._metadata[key] = value
    
    def set_metadatas(self, metadata : Dict):
        self._metadata = {**self._metadata, **metadata}

    def get_metadata(self) -> Dict:
        return self._metadata

class MME_ExportType(Enum):
    MATLAB_MAT = auto()
    MME_FILE = auto()

def save_mme(file : str, record : MMEContainer, file_type : MME_ExportType = MME_ExportType.MME_FILE):
    def __save_as_mme(file : str, record : MMEContainer):
        if os.path.isfile(file):
            raise ValueError(f'{file} already exist!')
        with ZipFile(file, 'w') as zf:
            # Save metadata
            zf.writestr('metadata.json', json.dumps(record.get_metadata(), indent = 4))
            # Save Lookup
            entities = record.get_entities()
            lookup = ''
            for e in entities:
                key = str(e.type)
                fname = os.path.basename(e.file)
                lookup += f'{key}={fname}\n'
                # Write files
                zf.write(e.file, os.path.basename(fname), compress_type=zipfile.ZIP_DEFLATED)
            # Save lookup file
            zf.writestr('lookup.info', lookup)

    def __save_as_mat(file : str, record : MMEContainer):
        if os.path.isfile(file):
            raise ValueError(f'{file} already exist!')

        lookup = {}
        data = {}
        for e in record.get_entities():
            fname = os.path.basename(e.file)
            lookup[str(e.type)] = fname
            data[str(e.type)] = e.data
        
        mat = {
            'metadata' : record.get_metadata(),
            'lookup' : lookup,
            'data' : data
        }
        savemat(file, mat, do_compression=True)
    
    {
        MME_ExportType.MATLAB_MAT: __save_as_mat,
        MME_ExportType.MME_FILE : __save_as_mme
    }[file_type](file, record)

class VTD_DatasetLoader(object):

    __thermal_dir__ = 'thermal'
    __visible_dir__ = 'visible'
    __depth_dir__ = 'depth'

    def __init__(self, root_dir : str) -> None:
        self._rootdir = root_dir
        self.init()

    @property
    def multimodal_dir(self) -> str:
        return self._rootdir
    
    def init(self) -> None:
        # Check the validity of directory
        if self._rootdir is None or not os.path.isdir(self._rootdir):
            raise ValueError(f'{self._rootdir} is an invalid directory path!')
        # dir_list = [os.path.join(self._rootdir, dname) for dname in os.listdir(self._rootdir) if os.path.isdir(self._rootdir)]
        self._thermal_dir = os.path.join(self._rootdir, self.__thermal_dir__)
        if not os.path.isdir(self._thermal_dir):
            raise ValueError('Thermal directory does not exist!')
        self._visible_dir = os.path.join(self._rootdir, self.__visible_dir__)
        if not os.path.isdir(self._visible_dir):
            raise ValueError('Visible directory does not exist!')
        self._depth_dir = os.path.join(self._rootdir, self.__depth_dir__)
        if not os.path.isdir(self._depth_dir):
            raise ValueError('Depth directory does not exist!')

    def generate_mme(self, file_type : MME_ExportType = MME_ExportType.MME_FILE):
        vfiles = glob.glob(os.path.join(self._visible_dir, '*.png'))
        vfiles.sort(key=os.path.getmtime)

        # Create the result directory
        res_dir = None
        file_extension = None
        if file_type == MME_ExportType.MME_FILE:
            res_dir = os.path.join(self._rootdir, 'mme')
            file_extension = 'mme'
        elif file_type == MME_ExportType.MATLAB_MAT:
            res_dir = os.path.join(self._rootdir, 'mat')
            file_extension = 'mat'
        else:
            raise ValueError(f'{file_type} does not supported!')

        Path(res_dir).mkdir(parents=True, exist_ok=True)

        for vf in vfiles:
            fname = os.path.basename(vf)
            ptn = re.findall('\d{12}\d+', fname)
            if not ptn:
                logging.warning(f'{fname} does not follow the supported naming!')
                continue
            ptn = ptn[0]
            
            tfname = f'thermal_{ptn}.png'
            dfname = f'depth_{ptn}.png'
            tf = os.path.join(self._thermal_dir, tfname)
            df = os.path.join(self._depth_dir, dfname)
            if not os.path.isfile(tf) or not os.path.isfile(df):
                logging.warning(f'{tfname} or {dfname} does not exist!')
                continue
            
            container = MMEContainer()
            container.add_entity(MMEntityType.Visible, vf)
            container.add_entity(MMEntityType.Thermal, tf)
            container.add_entity(MMEntityType.DepthMap, df)

            save_mme(
                os.path.join(res_dir, f'{ptn}.{file_extension}'),
                record=container,
                file_type=file_type
            )

            