"""Handle SPIM data"""
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib  import gridspec
from matplotlib.widgets import Slider


def plot_stack(stack):
    """Display stack with a slider to select the slice"""
    img_height = 8
    slider_height = 0.02
    width = img_height*stack.shape[1]/stack.shape[2]
    plt.figure(figsize=(img_height + slider_height, width))
    img_ax = plt.axes([0, 0.05, 1, 1])
    slider_ax = plt.axes([0.1, 0.014, 0.8, slider_height])
    mid_slice = stack.shape[0]//2 - 1
    img = img_ax.imshow(stack[mid_slice], cmap='Greys_r')
    img_ax.set_xticks([])
    img_ax.set_yticks([])

    slider = Slider(slider_ax, 'Slice', 0, stack.shape[0] - 1,
        valinit=mid_slice, color='#AAAAAA')
    slider.on_changed(lambda slice: img.set_data(stack[slice]))

    plt.show()


if __name__ == "__main__":
    """Open & display example stack"""
    # stack = io.imread('Examples/SPIM_example.tif')
    stack = np.random.rand(100,100,75)
    plot_stack(stack)


    """Aggregate excel files from SPIM analysis with MATLAB"""
    # DCs = pd.read_excel('../Data/SPIM_OutputExample/20k-1_BMDC_coordinates.xls',
    #     sheetname='Position', skiprows=1)[['Position X', 'Position Y', 'Position Z']]
    # DCs.columns = ['X', 'Y', 'Z']
    # DCs['Distance to Closest HEV'] = pd.read_excel('../Data/SPIM_OutputExample/20k-1_DC-HEV.xls',
    #     sheetname='Distances_to_closest_HEV', header=None)
    # DCs['Distance to LN Center'] = pd.read_excel('../Data/SPIM_OutputExample/20k-1_DC-HEV.xls',
    #     sheetname='Distances_to_Center_of_LN', header=None)
    # DCs['Cell Type'] = 'Dendritic Cell'
    #
    # T_cells = pd.read_excel('../Data/SPIM_OutputExample/20k-1_OT-I_coordinates.xls',
    #     sheetname='Position', skiprows=1)[['Position X', 'Position Y', 'Position Z']]
    # T_cells.columns = ['X', 'Y', 'Z']
    # T_cells['Distance to Closest HEV'] = pd.read_excel('../Data/SPIM_OutputExample/20k-1_T-HEV.xls',
    #     sheetname='Distances_to_closest_HEV', header=None)
    # T_cells['Distance to LN Center'] = pd.read_excel('../Data/SPIM_OutputExample/20k-1_T-HEV.xls',
    #     sheetname='Distances_to_Center_of_LN', header=None)
    # T_cells['Distance to Closest DC'] = pd.read_excel('../Data/SPIM_OutputExample/20k-1_T-DC.xls',
    #     sheetname='Distances_between_cell_types', header=None)
    # T_cells['Cell Type'] = 'T Cell'
    #
    # positions = DCs.append(T_cells)
    # print(positions)
    #
    # df = positions[['Cell Type', 'Distance to Closest HEV', 'Distance to LN Center']]
    # df.set_index([df.index, 'Cell Type']).unstack('Cell Type').swaplevel(0,1,axis = 1).sort(axis = 1)
    # print(df)
