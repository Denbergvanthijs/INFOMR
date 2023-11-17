import streamlit.components.v1 as components
from obj2html import obj2html

camera = {"fov": 45,
          "aspect": 2,
          "near": 0.1,
          "far": 100,
          "pos_x": 0,
          "pos_y": 10,
          "pos_z": 20,
          "orbit_x": 0,
          "orbit_y": 5,
          "orbit_z": 0}

light = {"color": "0x000000",
         "intensity": 1,
         "pos_x": 0,
         "pos_y": 10,
         "pos_z": 0,
         "target_x": -5,
         "target_y": 0,
         "target_z": 0}
obj_options = {"scale_x": 30,
               "scale_y": 30,
               "scale_z": 30}

html_string = obj2html('temp_mesh.obj', output_html_path=None, camera=camera,
                       light=light, obj_options=obj_options, html_elements_only=True)
components.html(html_string)
