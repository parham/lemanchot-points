
import phm.pipeline as pip

obj = pip.PHM_ICP_Pipeline(
    root_dir='/home/phm/GoogleDrive/Personal/Datasets/my-dataset/multi-modal/20210722_pipe_heating/vtd',
    filenames= [
        'vtd_1626967963384.mat', 'vtd_1626967965865.mat', 'vtd_1626967973439.mat'
    ]
)

obj.execute()