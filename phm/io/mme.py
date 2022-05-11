
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
from phm.io.modality import load_entity, supported_modality_loaders

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

    print(f'Total : {len(file_ids)}, Matched : {matched}')

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
    pass

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
