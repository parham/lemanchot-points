
from phm.data.core import MMEntityType
from phm.data.data import MME_ExportType, MMEContainer, VTD_DatasetLoader, save_mme

container = MMEContainer()
container.set_metadatas({
    'name' : 'Parham',
    'family' : 'Nooralishahi',
    'age' : 34,
    'gpa' : 4.22
})
container.add_entity(
    type=MMEntityType.Visible,
    file='/home/phm/Datasets/multi-modal/20210706_multi_modal/visible/visible_1625604430816.png'
)
container.add_entity(
    type=MMEntityType.Thermal,
    file='/home/phm/Datasets/multi-modal/20210706_multi_modal/thermal/thermal_1625604430816.png'
)
container.add_entity(
    type=MMEntityType.DepthMap,
    file='/home/phm/Datasets/multi-modal/20210706_multi_modal/depth/depth_1625604430816.png'
)

save_mme('test.mme', container, file_type=MME_ExportType.MME_FILE)
save_mme('test.mat', container, file_type=MME_ExportType.MATLAB_MAT)

