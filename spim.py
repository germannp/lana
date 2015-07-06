"""Handle SPIM data"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import data, filters, measure


def plot_stack(stack, cells=None):
    """Display stack with a slider to select the slice"""
    img_height = 8
    slider_height = 0.02
    width = img_height*stack.shape[1]/stack.shape[2]
    mid_slice = stack.shape[0]//2 - 1
    plt.figure(figsize=(img_height + slider_height, width))
    img_ax = plt.axes([0, 0.05, 1, 1])
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    img = img_ax.imshow(stack[mid_slice], interpolation='nearest',
        cmap='Greys_r')
    slider_ax = plt.axes([0.1, 0.014, 0.8, slider_height])
    slider = Slider(slider_ax, 'Slice', 0, stack.shape[0] - 1,
        valinit=mid_slice, valfmt=u'%d')
    slider.vline.set_color('k')
    slider.on_changed(lambda slice: img.set_data(stack[slice]))
    if cells is not None:
        for _, cell in cells.iterrows():
            img_ax.add_artist(plt.Circle((cell['X'], cell['Y']),
                cell['Volume']**(1/3), color='b', linewidth=2, fill=False))
    plt.show()


def find_cells(stack):
    """Label spots"""
    threshold = (stack.max()/10)
    labels = measure.label(stack > threshold)
    cells = pd.DataFrame()
    for label in np.unique(labels):
        locations = np.where(labels == label)
        cells.loc[label, 'X'] = np.mean(locations[2])
        cells.loc[label, 'Y'] = np.mean(locations[1])
        cells.loc[label, 'Z'] = np.mean(locations[0])
        cells.loc[label, 'Mean intensity'] = np.mean(stack[labels == label])
        cells.loc[label, 'Max. intensity'] = np.max(stack[labels == label])
        cells.loc[label, 'Volume'] = np.sum(labels == label)
    return cells[1:]


if __name__ == "__main__":
    """Open & display example stack"""
    # stack = data.imread('Examples/SPIM_example.tif')
    # plot_stack(stack)


    """Mock and find some cells"""
    stack = np.zeros((50, 200, 150))
    points = (stack.shape*np.random.rand(8, 3)).T.astype(np.int)
    stack[[point for point in points]] = 1
    stack = filters.gaussian_filter(stack, 1)
    cells = find_cells(stack)
    print(cells)
    plot_stack(stack, cells)


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
