
from phm.data.data import MME_ExportType, VTD_DatasetLoader


loader = VTD_DatasetLoader('/home/phm/Datasets/multi-modal/20210722_pipe_noheating')
loader.generate_mme(file_type=MME_ExportType.MATLAB_MAT)
loader.generate_mme(file_type=MME_ExportType.MME_FILE)