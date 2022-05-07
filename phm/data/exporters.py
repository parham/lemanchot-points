
import os
import json

from scipy.io import savemat
from zipfile import ZipFile, ZIP_DEFLATED
from phm.data.data import MMEContainer, mme_exporter

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
            zf.write(e.file, os.path.basename(fname), compress_type=ZIP_DEFLATED)
        # Save lookup file
        zf.writestr('lookup.inf', lookup)

@mme_exporter('mat')
def save_as_mat(file : str, record : MMEContainer, file_type : str):
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