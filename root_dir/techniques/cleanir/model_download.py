import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from cleanir.cleanir import Cleanir
from cleanir.tools.crop_face import *
from cleanir.tools.get_model import *
#import sys
#print sys.path
MODEL_PATH = './model'
get_cleanir_model(MODEL_PATH)
