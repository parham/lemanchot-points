

from ast import Sub
import os

from consolemenu import *
from consolemenu.items import *
from consolemenu.format import *

from colors import color
from dotmap import DotMap
from configparser import ConfigParser

from phm.dataset import create_dual_point_cloud_dataset, create_mme_dataset, create_point_cloud_dataset, create_vtd_dataset

class CLI_Tool:
    def __init__(self) -> None:
        self.settings = DotMap()
        self.settings.root_dir = os.getcwd()
    
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

        self.settings.root_dir = None
        rdir = input('Enter new root directory >> ')
        if not os.path.isdir(rdir):
            print(color('The directory does not exist!', fg='red'))
            return
        
        self.settings.root_dir = rdir

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

    def on_visualize_mme_data(self):
        pass

    def on_visualize_pc_data(self):
        pass

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
        
        menu_load_settings = FunctionItem("Load Settings", self.on_load_settings)
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
        menu_viz_mme_data = FunctionItem("Visualize Multi-modal Data (MME)", self.on_visualize_mme_data)
        submenu_visualize.append_item(menu_viz_mme_data)

        menu_viz_pc_data = FunctionItem("Visualize Multi-modal Point Cloud", self.on_visualize_pc_data)
        submenu_visualize.append_item(menu_viz_pc_data)

        submenu_visualize_item = SubmenuItem('Visualize Multi-modal Data', submenu=submenu_create)
        submenu_visualize_item.set_menu(menu)
        menu.append_item(submenu_visualize_item)

        # Create "Process Multi-modal Data" submenu
        submenu_process = ConsoleMenu(title='Process Multi-modal Data', exit_option_text='Back to main menu')

        menu.show()

def main():
    cli = CLI_Tool()
    cli.run()

if __name__ == '__main__':
    main()