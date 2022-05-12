
from typing import Any, Dict, Union
from dataclasses import dataclass

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
                self.add_entity(e)

    def add_entity(self, entity : MMERecord, overwrite : bool = False):
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
