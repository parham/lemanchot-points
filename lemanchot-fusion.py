

import os
import glob
import logging
import sys
import csv

import open3d as o3d
import open3d.visualization.gui as gui

from consolemenu import *
from consolemenu.items import *
from consolemenu.format import *

from colors import color
from dotmap import DotMap
from configparser import ConfigParser

from phm.dataset import VTD_Dataset, create_dual_point_cloud_dataset, create_mme_dataset, create_point_cloud_dataset, create_vtd_dataset
from phm.io.vtd import load_RGBDnT
from phm.pipeline.core import ConvertToPC_Step, FilterDepthRange_Step, Pipeline, PointCloudSaver_Step, RGBDnTBatch
from phm.pipeline.o3d_pipeline import ColoredICPRegistar_Step, O3DRegistrationMetrics_Step
from phm.pipeline.manual_pipeline import ManualRegistration_Step
from phm.pipeline.probreg_pipeline import CPDRegistration_Step, FilterregRegistration_Step, GMMTreeRegistration_Step, SVRRegistration_Step
from phm.visualization import VTD_Visualization, visualize_vtd
from phm.vtd import load_pinhole

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("system.log"), logging.StreamHandler(sys.stdout)],
)

class CLI_Tool:
    def __init__(self, root_dir : str = None) -> None:
        self.settings = DotMap()
        self.set_root_dir(root_dir if root_dir is not None else os.getcwd())
    
    def set_root_dir(self, root_dir : str):
        if not os.path.isdir(root_dir):
            msg = 'The directory does not exist!'
            print(color(msg, fg='red'))
            raise FileNotFoundError(msg)
        
        self.settings.root_dir = root_dir
        self.on_load_settings()
    
    @property
    def menu_format(self):
        menu_format = MenuFormatBuilder()
        menu_format.set_border_style_type(MenuBorderStyleType.DOUBLE_LINE_BORDER)
        menu_format.set_title_align('center') 
        menu_format.show_prologue_bottom_border(True)
        menu_format.show_prologue_top_border(True)
        return menu_format
    
    @property
    def tool_discription(self):
        return '''
In the rise of recent advancements in unmanned aerial vehicles, many studies have focused on using multi-modal platforms for remote inspection of industrial and construction sites. The acquisition of multiple data modalities assists the inspectors in acquiring comprehensive information about the surrounding environment and the targeted components.
LeManchot-Fusion is a CLI tool for fusion and processing of multi-modal data (especially which acquired by LeManchot-DC system). This software make the user able to read, convert and create multi-modal data in different levels. The system also provides required features to fuse the modalities of interest. Moreover, multiple built-in features has been implemented for modality alignment, noise reduction, and registration.

Contributors: 
    - Parham Nooralishahi <parham.nooralishahi.1@ulaval.ca>
    - Sandra Pozzer <sandra.pozzer.1@ulaval.ca>
    - Gabriel Ramos <gabriel.ramos.1@ulaval.ca>
Organization: Université Laval
Repository: https://github.com/parham/lemanchot-fusion
        '''

    def on_set_root_dir(self):
        print(f'Current root directory >> {self.settings.root_dir}')
        rdir = input('Enter new root directory >> ')
        self.set_root_dir(rdir)

    def on_load_settings(self):
        __info_file = os.path.join(self.settings.root_dir, 'ds.ini')
        if not os.path.isfile(__info_file):
            print(color('ds.inf', fg='red'))
            return
        self.__load_init_settings(__info_file)
    
    def on_create_mme_dataset(self):
        rgbdt_dir = os.path.join(self.settings.root_dir, self.settings.modalities.rgbdt_dir)
        create_mme_dataset(
            root_dir = self.settings.root_dir,
            res_dir = rgbdt_dir,
            file_type = 'mat'
        )
    
    def on_create_vtd_dataset(self):
        rgbdt_dir = os.path.join(self.settings.root_dir, self.settings.modalities.rgbdt_dir)
        vdt_dir = os.path.join(self.settings.root_dir, self.settings.modalities.vtd_dir)
        depth_param_file = os.path.join(self.settings.root_dir, self.settings.modalities.depth_param_file)
        h_fid = self.settings.calibration.calib_ref_id if \
            'calib_ref_id' in self.settings.calibration else None
        create_vtd_dataset(
            in_dir=rgbdt_dir,
            target_dir=vdt_dir,
            depth_param_file=depth_param_file,
            in_type='mat', 
            homography_fid=h_fid
        )
    
    def on_create_point_cloud_dataset(self):
        vdt_dir = os.path.join(self.settings.root_dir, self.settings.modalities.vtd_dir)
        pc_dir = os.path.join(self.settings.root_dir, self.settings.modalities.point_cloud_dir)
        create_point_cloud_dataset(
            in_dir = vdt_dir,
            target_dir = pc_dir,
            file_type='ply_txt'
        )

    def on_create_dual_point_cloud_dataset(self):
        vdt_dir = os.path.join(self.settings.root_dir, self.settings.modalities.vtd_dir)
        pc_dir = os.path.join(self.settings.root_dir, self.settings.modalities.dual_point_cloud_dir)
        create_dual_point_cloud_dataset(
            in_dir = vdt_dir,
            target_dir = pc_dir
        )

    def on_visualize_mm_point_cloud(self):
        pc_dir = os.path.join(self.settings.root_dir, self.settings.modalities.point_cloud_dir)
        fs = glob.glob(os.path.join(pc_dir, '*.ply'))
        for f in fs:
            print(os.path.basename(f))
        pc_file = input('Enter the file name >> ')
        pc_file = os.path.join(pc_dir, pc_file)
        if not os.path.isfile(pc_file):
            print(f'Invalid Filename! ({pc_file})')
            return
        # Visualization of o3d point cloud
        pcd = o3d.io.read_point_cloud(pc_file)
        o3d.visualization.draw_geometries([pcd])

    def on_visualize_vtd_data(self):
        pc_dir = os.path.join(self.settings.root_dir, self.settings.modalities.vtd_dir)
        fs = glob.glob(os.path.join(pc_dir, '*.mat'))
        for f in fs:
            print(os.path.basename(f))
        pc_file = input('Enter the file name >> ')
        pc_file = os.path.join(pc_dir, pc_file)
        if not os.path.isfile(pc_file):
            print(f'Invalid Filename! ({pc_file})')
            return
        # Visualization of VTD data
        pinhole = load_pinhole(os.path.join(self.settings.root_dir, self.settings.modalities.depth_param_file))
        data = load_RGBDnT(pc_file)
        gui.Application.instance.initialize()
        w = VTD_Visualization(data, 
            pinhole, 'PHM RGBD&T Visualization', 1024, 768)
        gui.Application.instance.run()
        print('VTD visualization is finished!')

    def _get_pipeline(self, 
        method_name : str,
        final_result_dir : str,
        aligned_result_dir : str,
        depth_param_file : str,
        depth_param
    ):        
        filter_depth = FilterDepthRange_Step()
        convert2pc = ConvertToPC_Step(
            depth_params_file = depth_param_file,
            data_batch_key = 'prp_frames')
        aligned_pc_saver = PointCloudSaver_Step(
            data_pcs_key='aligned_pcs',
            depth_param=depth_param,
            result_dir=aligned_result_dir,
            method_name=method_name
        )
        fused_pc_saver = PointCloudSaver_Step(
            data_pcs_key='fused_pc',
            depth_param=depth_param,
            result_dir=final_result_dir,
            method_name=method_name
        )
        metrics_step = O3DRegistrationMetrics_Step(data_pcs_key='pcs')
        if method_name == 'filterreg':
            return Pipeline([
                filter_depth, convert2pc,
                FilterregRegistration_Step(
                    voxel_size = 0.05,
                    data_pcs_key='pcs', maxiter=40),
                # ColoredICPRegistar_Step(data_pcs_key='pcs', max_iter=[1, 1, 1]),
                aligned_pc_saver, fused_pc_saver, metrics_step
            ])
        elif method_name == 'gmmtree':
            return Pipeline([
                filter_depth, convert2pc,
                GMMTreeRegistration_Step(
                    voxel_size = 0.05,
                    data_pcs_key='pcs', maxiter=40),
                ColoredICPRegistar_Step(data_pcs_key='pcs', max_iter=[1, 1, 1]),
                aligned_pc_saver, fused_pc_saver, metrics_step
            ])
        elif method_name == 'svr':
            return Pipeline([
                filter_depth, convert2pc,
                SVRRegistration_Step(
                    voxel_size = 0.05,
                    data_pcs_key='pcs', maxiter=40),
                ColoredICPRegistar_Step(data_pcs_key='pcs', max_iter=[1, 1, 1]),
                aligned_pc_saver, fused_pc_saver, metrics_step
            ])
        elif method_name == 'cpd':
            return Pipeline([
                filter_depth, convert2pc,
                CPDRegistration_Step(
                    voxel_size = 0.05,
                    data_pcs_key='pcs'),
                ColoredICPRegistar_Step(data_pcs_key='pcs', max_iter=[1, 1, 1]),
                aligned_pc_saver, fused_pc_saver, metrics_step
            ])
        elif method_name == 'manual':
            return Pipeline([
                filter_depth, convert2pc,
                ManualRegistration_Step(
                    depth_params=depth_param,
                    data_pcs_key='pcs'),
                ColoredICPRegistar_Step(data_pcs_key='pcs', max_iter=[1, 1, 1]),
                aligned_pc_saver, fused_pc_saver, metrics_step
            ])
        elif method_name == 'colored_icp':
            return Pipeline([
                filter_depth, convert2pc,
                ColoredICPRegistar_Step(data_pcs_key='pcs'),
                aligned_pc_saver, fused_pc_saver, metrics_step
            ])
        else:
            raise NotImplementedError(f'{method_name} is not supported!')
    
    def on_process_registration(self, method_name):
        # method_name = 'filterreg'
        root_dir = self.settings.root_dir
        final_result_dir = os.path.join(root_dir, 'results', 'final_pcs')
        aligned_result_dir = os.path.join(root_dir, 'results', 'aligned_pcs', method_name)
        vtd_dir = os.path.join(root_dir, 'vtd')
        depth_param_file = os.path.join(root_dir, 'depth/camera_info.json')
        depth_param = load_pinhole(depth_param_file)
        # Determine the VTD files
        # Input filenames
        vtd_files = glob.glob(os.path.join(vtd_dir,'*_*.mat'))
        vtd_files = [os.path.basename(f) for f in vtd_files]
        for i in range(0, len(vtd_files), 3):
            chunk = vtd_files[i:i + 3]
            print(*chunk, sep = '\t')
        instr = input('Choose the filenames (seperated by comma) >> ')
        vtd_files = instr.split(',')

        # Loading Dataset
        batch = RGBDnTBatch(
            root_dir = vtd_dir,
            filenames = vtd_files
        )

        # Create the processing pipeline
        pipobj = self._get_pipeline(method_name,
            final_result_dir, aligned_result_dir,
            depth_param_file, depth_param)
        # Apply the pipeline on loaded data
        res = pipobj(batch)
        res_pc = res['fused_pc']
        metrics = res['metrics']

        self.__save_metrics(final_result_dir, method_name, metrics)
        visualize_vtd(
            res_pc, depth_param,
            f'Result of {method_name} technique', 1024, 768)

    def __save_metrics(self, result_dir, method_name, metrics):
        with open(os.path.join(result_dir, f'{method_name}_metrics.csv'), 'w', newline='') as csvfile:
            keys = tuple(metrics.keys())

            records = []
            for index in range(len(metrics[keys[0]])):
                tmp = {}
                for k in keys:
                    tmp[k] = metrics[k][index]
                records.append(tmp)

            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    def __load_init_settings(self, fs):
        config = ConfigParser()
        config.read(fs)
        for sec in config.sections():
            for key in config[sec]:
                self.settings[sec][key] = config[sec][key]
        print(config)

    def run(self):
        # Create main menu
        menu = ConsoleMenu(color("LeManchot-Fusion", fg='blue'),
            "The toolbox for fusion and processing of multi-modal data collected by LeManchot-DC system.",
            prologue_text=self.tool_discription,
            formatter=self.menu_format)

        menu_set_root_dir = FunctionItem("Set/Change Root Directory", self.on_set_root_dir)
        menu.append_item(menu_set_root_dir)
        
        menu_load_settings = FunctionItem("Reload Settings", self.on_load_settings)
        menu.append_item(menu_load_settings)
        
        # Create "Create Multi-modal data" submenu
        submenu_create = ConsoleMenu(title='Create Multi-modal Data', exit_option_text='Back to main menu')
        
        menu_create_mme_dataset = FunctionItem("Create RGBD&T Dataset (MME)", self.on_create_mme_dataset)
        submenu_create.append_item(menu_create_mme_dataset)
        
        menu_create_vtd_dataset = FunctionItem("Create VTD Dataset", self.on_create_vtd_dataset)
        submenu_create.append_item(menu_create_vtd_dataset)
        
        menu_create_pc_dataset = FunctionItem("Create Point Cloud Dataset", self.on_create_point_cloud_dataset)
        submenu_create.append_item(menu_create_pc_dataset)
        
        menu_create_dual_pc_dataset = FunctionItem("Create Dual Point Cloud Dataset", self.on_create_dual_point_cloud_dataset)
        submenu_create.append_item(menu_create_dual_pc_dataset)

        submenu_create_item = SubmenuItem('Create Multi-modal Data', submenu=submenu_create)
        submenu_create_item.set_menu(menu)
        menu.append_item(submenu_create_item)

        # Create "Visualize Multi-modal Data" submenu
        submenu_visualize = ConsoleMenu(title='Visualize Multi-modal Data', exit_option_text='Back to main menu')
        menu_viz_mme_data = FunctionItem("Visualize Multi-modal Data (MME)", self.on_visualize_vtd_data)
        submenu_visualize.append_item(menu_viz_mme_data)

        menu_viz_pc_data = FunctionItem("Visualize Multi-modal Point Cloud", self.on_visualize_mm_point_cloud)
        submenu_visualize.append_item(menu_viz_pc_data)

        submenu_visualize_item = SubmenuItem('Visualize Multi-modal Data', submenu=submenu_visualize)
        submenu_visualize_item.set_menu(menu)
        menu.append_item(submenu_visualize_item)

        # Create "Process Multi-modal Data" submenu
        submenu_process = ConsoleMenu(title='Register Multi-modal Point Cloud', exit_option_text='Back to main menu')

        menu_process_filterreg = FunctionItem("Multi-modal Registration using FilterReg", lambda: self.on_process_registration('filterreg'))
        submenu_process.append_item(menu_process_filterreg)

        menu_process_gmtree = FunctionItem("Multi-modal Registration using GMMTree", lambda: self.on_process_registration('gmmtree'))
        submenu_process.append_item(menu_process_gmtree)

        menu_process_svr = FunctionItem("Multi-modal Registration using SVR", lambda: self.on_process_registration('svr'))
        submenu_process.append_item(menu_process_svr)

        menu_process_cpd = FunctionItem("Multi-modal Registration using CPD", lambda: self.on_process_registration('cpd'))
        submenu_process.append_item(menu_process_cpd)

        menu_process_manual = FunctionItem("Multi-modal Registration Manually", lambda: self.on_process_registration('manual'))
        submenu_process.append_item(menu_process_manual)

        menu_process_colored_icp = FunctionItem("Multi-modal Registration using Colored ICP", lambda: self.on_process_registration('colored_icp'))
        submenu_process.append_item(menu_process_colored_icp)

        submenu_process_item = SubmenuItem('Register Multi-modal Point Cloud', submenu=submenu_process)
        submenu_process_item.set_menu(menu)
        menu.append_item(submenu_process_item)

        menu.show()

def main():

    args = sys.argv
    if len(args) != 2:
        print('Invalid arguments has been detected!')

    cli = CLI_Tool(root_dir=args[1])
    cli.run()

if __name__ == '__main__':
    main()