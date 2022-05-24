

import os

from consolemenu import *
from consolemenu.items import *
from consolemenu.format import *

from colors import color
from dotmap import DotMap
from configparser import ConfigParser

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
        if os.path.isfile(__info_file):
            print(color('ds.inf', fg='red'))
            return
        self.__load_init_settings(__info_file)
    
    def __load_init_settings(self, fs):
        config = ConfigParser()
        config.read(fs)
        for sec in config.sections():
            for key in config[sec]:
                self.settings[sec][key] = config[sec][key]

    def run(self):
        menu = ConsoleMenu(color("LeManchot-Fusion", fg='blue'),
            "The toolbox for fusion and processing of multi-modal data collected by LeManchot-DC system.",
            prologue_text=self.tool_discription,
            formatter=self.menu_format)

        menu_set_root_dir = FunctionItem("Set/Change Root Directory ", self.on_set_root_dir)
        menu_load_settings = FunctionItem("Load Settings", self.on_load_settings)
        menu.append_item(menu_set_root_dir)
        menu.append_item(menu_load_settings)
        menu.show()

def main():
    cli = CLI_Tool()
    cli.run()

if __name__ == '__main__':
    main()