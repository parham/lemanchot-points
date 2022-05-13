
import os
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from dotmap import DotMap
from enum import Enum, unique

from phm.data import RGBDnT_O3D

@unique
class Modalities(Enum):
    Visible_Depth = 0,
    Thermal_Depth = 1,
    Depth_Only = 2,

    def __str__(self):
        return self.label

    @property
    def label(self):
        return {
            Modalities.Visible_Depth : 'Visible Depth',
            Modalities.Thermal_Depth : 'Thermal Depth',
            Modalities.Depth_Only : 'Depth Only'
        }[self]

class VTD_Visualization:

    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    def __init__(self, 
        data : RGBD
        win_width : int = 200, 
        win_height : int = 200
    ):
        # Initialize the settings
        self.__init_settings()
        # Create the window instance
        self.window = gui.Application.instance.create_window("Open3D", win_width, win_height)
        # Create the 3D Widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        # Setting the relative measurements
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))
        # Create the setting panel
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        ############# View Control Panel
        view_ctrls = gui.CollapsableVert(
            "View Controls", 0.25 * em, gui.Margins(em, 0, 0, 0))
        # Show Skymap (Checkbox)
        self._show_skybox = gui.Checkbox("Show Skymap")
        self._show_skybox.set_on_checked(self._on_show_skybox)
        view_ctrls.add_child(self._show_skybox)

        view_ctrls.add_fixed(separation_height)
        grid = gui.VGrid(2, 0.25 * em)
        # Select Background
        self._bg_color = gui.ColorEdit()
        self._bg_color.set_on_value_changed(self._on_bg_color)
        grid.add_child(gui.Label("Background"))
        grid.add_child(self._bg_color)
        # Select Point Cloud
        grid.add_child(gui.Label("Point size"))
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)
        grid.add_child(self._point_size)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)
        self._settings_panel.add_fixed(separation_height)
        ############# Modalities Panel
        modalities_ctrls = gui.CollapsableVert(
            "Modalities", 0.25 * em, gui.Margins(em, 0, 0, 0))
        grid = gui.VGrid(2, 0.25 * em)
        self._modalities = gui.Combobox()
        for m in Modalities:
            self._modalities.add_item(m.label)
        self._modalities.set_on_selection_changed(self._on_modalities)
        grid.add_child(gui.Label("Modalities"))
        grid.add_child(self._modalities)
        modalities_ctrls.add_child(grid)

        self._settings_panel.add_child(modalities_ctrls)
        self._settings_panel.add_fixed(separation_height)
        ############# Filter Panel
        filters_ctrls = gui.CollapsableVert(
            "Filters", 0.25 * em, gui.Margins(em, 0, 0, 0))

        self._settings_panel.add_child(filters_ctrls)
        self._settings_panel.add_fixed(separation_height)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._settings_panel)

        self.__init_menu()

    def __init_menu(self):
        # Application Menubar
        if gui.Application.instance.menubar is None:
            # File Menu
            file_menu = gui.Menu()
            file_menu.add_item("Open...", VTD_Visualization.MENU_OPEN)
            file_menu.add_item("Export Current Image...", VTD_Visualization.MENU_EXPORT)
            file_menu.add_separator()
            file_menu.add_item("Quit", VTD_Visualization.MENU_QUIT)
            # Settings Menu
            settings_menu = gui.Menu()
            settings_menu.add_item(
                "Settings Panel",
                VTD_Visualization.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(VTD_Visualization.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", VTD_Visualization.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Settings", settings_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

            self.window.set_on_menu_item_activated(
                VTD_Visualization.MENU_OPEN, 
                self._on_menu_open)
            self.window.set_on_menu_item_activated(
                VTD_Visualization.MENU_EXPORT,
                self._on_menu_export)
            self.window.set_on_menu_item_activated(
                VTD_Visualization.MENU_QUIT, 
                self._on_menu_quit)
            self.window.set_on_menu_item_activated(
                VTD_Visualization.MENU_SHOW_SETTINGS,
                self._on_menu_toggle_settings_panel)
            self.window.set_on_menu_item_activated(
                VTD_Visualization.MENU_ABOUT, 
                self._on_menu_about)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb", "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        def _on_load_dialog_done(self, filename):
            self.window.close_dialog()
            self.load(filename)

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, 
            "Choose file to save", self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_modalities(self, name, index):
        self.settings.modalities = Modalities(index)
        self._apply_settings()

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def __init_settings(self):
        self.settings = DotMap()

    def _apply_settings(self):
        # Skymap
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self._show_skybox.checked = self.settings.show_skybox
        # Background
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._bg_color.color_value = self.settings.bg_color
        # Point Cloud
        self._point_size.double_value = self.settings.material.point_size


        
        self._use_ibl.checked = self.settings.use_ibl
        self._use_sun.checked = self.settings.use_sun
        self._ibl_intensity.int_value = self.settings.ibl_intensity
        self._sun_intensity.int_value = self.settings.sun_intensity
        self._sun_dir.vector_value = self.settings.sun_dir
        self._sun_color.color_value = self.settings.sun_color
        self._material_prefab.enabled = (
            self.settings.material.shader == Settings.LIT)
        c = gui.Color(self.settings.material.base_color[0],
                      self.settings.material.base_color[1],
                      self.settings.material.base_color[2],
                      self.settings.material.base_color[3])
        self._material_color.color_value = c

if __name__ == "__main__":
    gui.Application.instance.initialize()

    w = VTD_Visualization(1024, 768)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error", "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()