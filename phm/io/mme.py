
import re
import os
import json
import logging
import glob

from pathlib import Path
from typing import List, Tuple, Union
from scipy.io import savemat, loadmat
from zipfile import ZipFile, ZIP_DEFLATED

from phm.data import MMEContainer, MMERecord
from phm.io.modality import supported_modality_loaders

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

@mme_exporter('mme')
def save_as_mme(file : str, record : MMEContainer, file_type : str):    
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
            zf.write(e.file, arcname=f'{e.type}.png', compress_type=ZIP_DEFLATED)
        # Save lookup file
        zf.writestr('lookup.inf', lookup)

@mme_loader('mme')
def load_mme_file(file : str, file_type : str):
    raise NotImplementedError()

@mme_exporter('mat')
def save_as_mat(file : str, record : MMEContainer, file_type : str):
    lookup = {}
    data = {}
    for e in record.get_entities():
        fname = os.path.basename(e.file)
        lookup[e.type] = fname
        data[e.type] = e.data
    
    mat = {
        'metadata' : record.get_metadata(),
        'lookup' : lookup,
        'cid' : record.container_id
    }
    mat = {**mat, **data}
    savemat(file, mat, do_compression=True)

@mme_loader('mat')
def load_mat_file(file : str, file_type : str):
    data = loadmat(file_name=file)
    cid = data['cid'] if 'cid' in data else ''
    obj = MMEContainer(cid=cid)
    # metadata
    # in case metadata is none, the loadmat returns a numpy array with None value, 
    # so that's why we use data['metadata'] != None
    if data['metadata'] != None:
        obj.set_metadatas(data['metadata'])
    types = supported_modality_loaders()
    for dtype in types:
        if not dtype in data:
            continue
        dt = data[dtype]
        obj.add_entity(MMERecord(
            file=f'{dtype}.png',
            data=dt,
            type=dtype
        ))
    return obj
