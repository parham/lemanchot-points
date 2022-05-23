
import os
import sys
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from dotmap import DotMap
from enum import Enum, unique

from phm.data import RGBDnT

@unique
class Modalities(Enum):
    Visible_Depth = 0,
    Thermal_Depth = 1

    def __str__(self):
        return self.label

    @staticmethod
    def from_index(index : int):
        items = list(Modalities)
        return items[index]

    @property
    def label(self):
        return {
            Modalities.Visible_Depth : 'Visible Depth',
            Modalities.Thermal_Depth : 'Thermal Depth'
        }[self]

class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    DEFAULT_PROFILE_NAME = "Bright day with sun at +Y [default]"
    POINT_CLOUD_PROFILE_NAME = "Cloudy day (no direct sun)"
    CUSTOM_PROFILE_NAME = "Custom"
    LIGHTING_PROFILES = {
        DEFAULT_PROFILE_NAME: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "Less Bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        POINT_CLOUD_PROFILE_NAME: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    DEFAULT_MATERIAL_NAME = "Polished ceramic [default]"
    PREFAB = {
        DEFAULT_MATERIAL_NAME: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0
        },
        "Plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0
        },
        "Glazed ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0
        },
        "Clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0
        },
    }

    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_skybox = False
        self.show_axes = False
        self.use_ibl = True
        self.use_sun = True
        self.new_ibl_name = None  # clear to None after loading
        self.ibl_intensity = 45000
        self.sun_intensity = 45000
        self.sun_dir = [0.577, -0.577, -0.577]
        self.sun_color = gui.Color(1, 1, 1)

        self.apply_material = True  # clear to False after processing
        self._materials = {
            Settings.LIT: rendering.MaterialRecord(),
            Settings.UNLIT: rendering.MaterialRecord(),
            Settings.NORMALS: rendering.MaterialRecord(),
            Settings.DEPTH: rendering.MaterialRecord()
        }
        self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.LIT].shader = Settings.LIT
        self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
        self._materials[Settings.UNLIT].shader = Settings.UNLIT
        self._materials[Settings.NORMALS].shader = Settings.NORMALS
        self._materials[Settings.DEPTH].shader = Settings.DEPTH

        # Conveniently, assigning from self._materials[...] assigns a reference,
        # not a copy, so if we change the property of a material, then switch
        # to another one, then come back, the old setting will still be there.
        self.material = self._materials[Settings.UNLIT]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert (self.material.shader == Settings.LIT)
        prefab = Settings.PREFAB[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = Settings.LIGHTING_PROFILES[name]
        for key, val in profile.items():
            setattr(self, key, val)

class VTD_Visualization:

    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    def __init__(self, 
        data : RGBDnT,
        pinhole,
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
        ############# View Control Panel #############
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
        ############# Modalities Panel #############
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
        ############# Filter Panel #############
        filters_ctrls = gui.CollapsableVert(
            "Filters", 0.25 * em, gui.Margins(em, 0, 0, 0))

        self._settings_panel.add_child(filters_ctrls)
        self._settings_panel.add_fixed(separation_height)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._settings_panel)

        self.__init_menu()

        self.__data = data
        self._pinhole = pinhole
        # self.set_point_cloud(self.__data.to_point_cloud_visible_o3d(pinhole))

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

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Settings", settings_menu)
            
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

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, 
            r.y, width, height)

    def _on_point_size(self, size):
        self.settings.material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            VTD_Visualization.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

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

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

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

    def load_visible(self):
        self.set_point_cloud(self.__data.to_point_cloud_visible_o3d(self._pinhole))
    
    def load_thermal(self):
        self.set_point_cloud(self.__data.to_point_cloud_thermal_o3d(self._pinhole))

    def set_point_cloud(self, pc):
        self._pcloud = pc
        self._scene.scene.clear_geometry()

        if self._pcloud is not None:
            if not self._pcloud.has_normals():
                self._pcloud.estimate_normals()
            self._pcloud.normalize_normals()
            try:
                self._scene.scene.add_geometry("__model__", 
                    self._pcloud, self.settings.material)
                bounds = self._pcloud.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)  
        else:
            print("[WARNING] Failed to load points")  


    def _on_modalities(self, name, index):
        self.settings.modalities = Modalities.from_index(index)
        if self.settings.modalities == Modalities.Visible_Depth:
            self.load_visible()
        elif self.settings.modalities == Modalities.Thermal_Depth:
            self.load_thermal()

        self._apply_settings()

    def _on_bg_color(self, new_color):
        self.settings.bg_color = new_color
        self._apply_settings()

    def _on_show_skybox(self, show):
        self.settings.show_skybox = show
        self._apply_settings()

    def __init_settings(self):
        self.settings = Settings()

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

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False

        self._scene.scene.clear_geometry()
