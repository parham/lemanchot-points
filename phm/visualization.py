
import copy
import os
import sys
from typing import List
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from dotmap import DotMap
from enum import Enum, unique
from phm.data import RGBDnT
from phm.data.vtd import O3DPointCloudWrapper
from phm.vtd import load_pinhole

def pick_points(
    data : O3DPointCloudWrapper,
    depth_params    
):
    return pick_points_point_cloud(data.to_point_cloud_visible_o3d(intrinsic=depth_params))

def pick_points_point_cloud(data):
    print("====================================================")
    print("Selecting Control Points in the given point cloud")
    print("Pick the corresponding points using [shift + left click] (at least three correspondences)")
    print("-- Press [shift + right click] to undo point picking")
    print("After picking points, press 'Q' to close the window")
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(data)
    vis.run()  # the window is executed to let user select the points
    vis.destroy_window()
    print("====================================================")
    points = vis.get_picked_points()
    print(f'Total Selected Points >> {len(points)}')
    return points

def visualize_pointclouds_with_transformations(pcs : List, transformations : List):
    if len(pcs) == 0 or len(transformations) == 0:
        raise ValueError('Pointclouds or transformations are not given!')
    if len(pcs) != len(transformations):
        raise ValueError('Pointclouds and transformations must have same size')
    
    res = []
    index = 0
    for index in range(len(pcs)):
        pc = pcs[index]
        tmp = copy.deepcopy(pc)
        tmp.transform(transformations[index])
        res.append(tmp)
        index += 1
    o3d.visualization.draw_geometries(res)

@unique
class Modalities(Enum):
    Visible_Depth = 0,
    Thermal_Depth = 1,
    Fusion = 2,
    Normals = 3

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
            Modalities.Thermal_Depth : 'Thermal Depth',
            Modalities.Fusion : 'Fusion',
            Modalities.Normals : 'Normals'
        }[self]

def visualize_vtd(
    data : O3DPointCloudWrapper,
    pinhole,
    win_name : str = 'VTD Visualization',
    win_width : int = 200, 
    win_height : int = 200
):
    gui.Application.instance.initialize()
    VTD_Visualization(data, pinhole, win_name, win_width, win_height)
    gui.Application.instance.run()

class VTD_Visualization:

    MENU_SCREENSHOT_EXPORT = 11
    MENU_QUIT = 12
    MENU_SHOW_SETTINGS = 21
    MENU_ABOUT = 31

    UNLIT = "defaultUnlit"
    LIT = "defaultLit"
    NORMALS = "normals"
    DEPTH = "depth"

    def __init__(self,
        data : O3DPointCloudWrapper,
        pinhole,
        win_name : str = 'VTD Visualization',
        win_width : int = 200, 
        win_height : int = 200
    ) -> None:
        self.__data = data
        self._pinhole = pinhole
        self.win_name = win_name
        self.win_width = win_width
        self.win_height = win_height
        self.__first_initialize = True
        # Initialize the settings
        self.__initialize_settings()
        # Initialize the window gui
        self.__initialize_window()
        # Initialize the events
        self.__initialize_events()
        # Apply Settings
        self.apply_settings()

    def __initialize_settings(self):
        self.settings = DotMap()
        # Set Background Color
        self.settings.background_color = gui.Color(1, 1, 1)
        self.settings.show_skybox = False
        self.settings.show_axes = False
        self.settings.apply_material = False
        # Initialize materials
        self.settings.materials.default = rendering.MaterialRecord()
        self.settings.materials.default.point_size = 1
        self.settings.materials.default.base_color = [0.9, 0.9, 0.9, 1.0]
        self.settings.materials.default.shader = 'defaultUnlit'
        # ####
        self.settings.modality = Modalities.Visible_Depth

    def __initialize_window(self):
        # Create the window instance
        self.window = gui.Application.instance.create_window(
            self.win_name, self.win_width, self.win_height)
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
        self.checkb_skybox = gui.Checkbox("Show Skymap")
        view_ctrls.add_child(self.checkb_skybox)
        view_ctrls.add_fixed(separation_height)
        #######
        grid = gui.VGrid(2, 0.25 * em)
        # Select Background Color
        self.background_color_edit = gui.ColorEdit()
        grid.add_child(gui.Label("Background"))
        grid.add_child(self.background_color_edit)
        # Select Point Cloud
        grid.add_child(gui.Label("Point Size"))
        self.psize_slider = gui.Slider(gui.Slider.INT)
        self.psize_slider.set_limits(1, 10)
        grid.add_child(self.psize_slider)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)
        self._settings_panel.add_fixed(separation_height)
        ############# Modalities Panel #############
        modalities_ctrls = gui.CollapsableVert(
            "Modalities", 0.25 * em, gui.Margins(em, 0, 0, 0))
        grid = gui.VGrid(2, 0.25 * em)
        self.modalities = gui.Combobox()
        for m in Modalities:
            self.modalities.add_item(m.label)
        grid.add_child(gui.Label("Modalities"))
        grid.add_child(self.modalities)
        modalities_ctrls.add_child(grid)

        self._settings_panel.add_child(modalities_ctrls)
        self._settings_panel.add_fixed(separation_height)
        ############# Filter Panel #############
        filters_ctrls = gui.CollapsableVert(
            "Filters", 0.25 * em, gui.Margins(em, 0, 0, 0))

        self._settings_panel.add_child(filters_ctrls)
        self._settings_panel.add_fixed(separation_height)

        self.window.add_child(self._scene)
        self.window.add_child(self._settings_panel)
        ############# Menu #############
        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        if gui.Application.instance.menubar is None:
            # File Menu
            file_menu = gui.Menu()
            file_menu.add_item("Export Current Image...", VTD_Visualization.MENU_SCREENSHOT_EXPORT)
            # Tools Menu
            tools_menu = gui.Menu()
            tools_menu.add_item("Settings Toolbar", VTD_Visualization.MENU_SHOW_SETTINGS)
            tools_menu.set_checked(VTD_Visualization.MENU_SHOW_SETTINGS, True)
            # Help Menu
            help_menu = gui.Menu()
            help_menu.add_item("About", VTD_Visualization.MENU_ABOUT)
            # Main Menu
            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Tools", tools_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

    def __initialize_events(self):
        def on_skybox_checkbox(show):
            self.settings.show_skybox = show
            self.apply_settings()

        def on_background_color(new_color):
            self.settings.background_color = new_color
            self.apply_settings()

        def on_point_size(size):
            self.settings.materials.default.point_size = int(size)
            self.settings.apply_material = True
            self.apply_settings()

        def on_modality_changed(name, index):
            self.settings.modality = Modalities.from_index(index)
            self.settings.apply_material = True
            self.apply_settings()

        def on_layout(layout_context):
            # The on_layout callback should set the frame (position + size) of every
            # child correctly. After the callback is done the window will layout
            # the grandchildren.
            r = self.window.content_rect
            self._scene.frame = r
            width = 17 * layout_context.theme.font_size
            height = min(r.height,
                self._settings_panel.calc_preferred_size(
                    layout_context, gui.Widget.Constraints()).height)
            self._settings_panel.frame = gui.Rect(r.get_right() - width, 
                r.y, width, height)

        def on_export_dialog_done(filename):
            self.window.close_dialog()
            frame = self._scene.frame
            self.export_image(filename, frame.width, frame.height)

        def on_menu_export():
            dlg = gui.FileDialog(
                gui.FileDialog.SAVE, 
                "Choose file to save",
                self.window.theme)
            dlg.add_filter(".png", "PNG files (.png)")
            dlg.set_on_cancel(lambda _: self.window.close_dialog())
            dlg.set_on_done(on_export_dialog_done)
            self.window.show_dialog(dlg)

        def on_menu_exit():
            gui.Application.instance.quit()

        def on_menu_toggle_settings_panel():
            self._settings_panel.visible = not self._settings_panel.visible
            gui.Application.instance.menubar.set_checked(
                VTD_Visualization.MENU_SHOW_SETTINGS, self._settings_panel.visible)

        def on_menu_about():
            # Show a simple dialog. Although the Dialog is actually a widget, you can
            # treat it similar to a Window for layout and put all the widgets in a
            # layout which you make the only child of the Dialog.
            em = self.window.theme.font_size
            dlg = gui.Dialog("About")

            # Add the text
            dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
            dlg_layout.add_child(gui.Label("LeManchot-Fusion"))

            # Add the Ok button. We need to define a callback function to handle
            # the click.
            ok = gui.Button("OK")
            ok.set_on_clicked(self.window.close_dialog)

            # We want the Ok button to be an the right side, so we need to add
            # a stretch item to the layout, otherwise the button will be the size
            # of the entire row. A stretch item takes up as much space as it can,
            # which forces the button to be its minimum size.
            h = gui.Horiz()
            h.add_stretch()
            h.add_child(ok)
            h.add_stretch()
            dlg_layout.add_child(h)

            dlg.add_child(dlg_layout)
            self.window.show_dialog(dlg)

        self.checkb_skybox.set_on_checked(on_skybox_checkbox)
        self.background_color_edit.set_on_value_changed(on_background_color)
        self.psize_slider.set_on_value_changed(on_point_size)
        self.modalities.set_on_selection_changed(on_modality_changed)
        self.window.set_on_layout(on_layout)
        # Menu Events
        self.window.set_on_menu_item_activated(VTD_Visualization.MENU_SCREENSHOT_EXPORT, on_menu_export)
        self.window.set_on_menu_item_activated(VTD_Visualization.MENU_QUIT, on_menu_exit)
        self.window.set_on_menu_item_activated(VTD_Visualization.MENU_SHOW_SETTINGS, on_menu_toggle_settings_panel)
        self.window.set_on_menu_item_activated(VTD_Visualization.MENU_ABOUT, on_menu_about)

    def apply_settings(self):
        # Render the Scene
        self.render()
        # Skymap
        self._scene.scene.show_skybox(self.settings.show_skybox)
        self.checkb_skybox.checked = self.settings.show_skybox
        # Background
        bg_color = [
            self.settings.background_color.red, 
            self.settings.background_color.green,
            self.settings.background_color.blue, 
            self.settings.background_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self.background_color_edit.color_value = self.settings.background_color
        # Point Cloud
        self.psize_slider.double_value = self.settings.materials.default.point_size

        if self.settings.apply_material:
            self._scene.scene.update_material(self.settings.materials.default)
            self.settings.apply_material = False

    def render(self):
        point_cloud = None
        if self.settings.modality == Modalities.Visible_Depth:
            point_cloud = self.__data.get_visible_point_cloud(
                intrinsic=self._pinhole, calc_normals=True)
            self.settings.materials.default.shader = VTD_Visualization.UNLIT
        elif self.settings.modality == Modalities.Thermal_Depth:
            point_cloud = self.__data.get_thermal_point_cloud(
                intrinsic=self._pinhole, calc_normals=True, remove_invalids=True)
            self.settings.materials.default.shader = VTD_Visualization.UNLIT
        elif self.settings.modality == Modalities.Normals:
            point_cloud = self.__data.get_visible_point_cloud(
                intrinsic=self._pinhole, calc_normals=True)
            self.settings.materials.default.shader = VTD_Visualization.NORMALS
        elif self.settings.modality == Modalities.Fusion:
            point_cloud = self.__data.get_fused_point_cloud(
                intrinsic=self._pinhole, calc_normals=True)
        else:
            raise NotImplemented('The modality is not implemented!')

        self._scene.scene.clear_geometry()
        try:
            self._scene.scene.add_geometry("__model__", 
                point_cloud, self.settings.materials.default)
            if self.__first_initialize:
                bounds = point_cloud.get_axis_aligned_bounding_box()
                self._scene.setup_camera(60, bounds, bounds.get_center())
                self.__first_initialize = False
        except Exception as e:
            print(e)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image
            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)
