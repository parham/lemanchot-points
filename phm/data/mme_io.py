
import os
import json

from scipy.io import savemat, loadmat
from zipfile import ZipFile, ZIP_DEFLATED
from phm.data.data import MMEContainer, MMERecord, mme_exporter, mme_loader, supported_mme_loaders, supported_modality_loaders

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
        obj.add_entity_record(MMERecord(dt, f'{dtype}.png', dtype))
    return obj