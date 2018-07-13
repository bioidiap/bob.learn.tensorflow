
from bob.learn.tensorflow.network import dummy
architecture = dummy
import pkg_resources
     
checkpoint_dir = "./temp/"

style_end_points = ["conv1"]
content_end_points = ["fc1"]

scopes = {"Dummy/":"Dummy/"}

