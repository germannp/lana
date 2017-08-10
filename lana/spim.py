"""Display and find cells in SPIM data"""
import psutil

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import data, filters, measure


def read_stack(path):
    """Load and return stack as normalized unsigned 8bit ints"""
    assert psutil.virtual_memory()[1] > 1000*1024*1024, \
        'Less than 1GiB of memory available'
    stack = data.imread(path)
    stack -= stack.min()
    np.true_divide(stack, stack.max() / 255, out=stack, casting='unsafe')
    return stack.astype(np.uint8)


def adjust_gamma(stack, gamma=0.5):
    """Return normalized stack**gamma"""
    assert psutil.virtual_memory()[1] > 1.5*stack.nbytes, \
        'Not enough memory available'
    scale = 255 / stack.max()**gamma
    for i, slice in enumerate(stack):
        slice = scale * slice**gamma
        stack[i] = slice.astype(np.uint8)
    return stack


def stacks2rgb(stack1, stack2, stack3=None):
    """Shape stacks for channels into RGB stack"""
    assert psutil.virtual_memory()[1] > 3*stack1.nbytes, \
        'Not enough memory for the RGB stack available'
    rgb_stack = np.zeros(stack1.shape + (3,), dtype=np.uint8)
    rgb_stack[:, :, :, 0] = stack1
    rgb_stack[:, :, :, 1] = stack2
    if stack3 is not None:
        rgb_stack[:, :, :, 2] = stack3
    return rgb_stack


def plot_stack(stack, cells=None):
    """Display stack with a slider to select the slice"""
    assert psutil.virtual_memory()[1] > stack.nbytes, \
        'Not enough memory available'
    img_height = 8
    width = img_height * stack.shape[2] / stack.shape[1]
    mid_slice = stack.shape[0] // 2 - 1
    plt.figure(figsize=(width, img_height * 1.05))
    img_ax = plt.axes([0, 0.05, 1, 0.95])
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    img = img_ax.imshow(stack[mid_slice], interpolation='none', cmap='gray')
    slider_ax = plt.axes([0.1, 0.014, 0.8, 0.02])
    slider = Slider(
        slider_ax, 'Slice', 0, stack.shape[0] - 1, valinit=mid_slice,
        valfmt=u'%d')
    slider.vline.set_color('k')
    slider.on_changed(lambda slice: img.set_data(stack[slice]))
    if cells is not None:
        for _, cell in cells.iterrows():
            img_ax.add_artist(
                plt.Circle((cell['X'], cell['Y']), cell['Volume']**(1 / 3),
                           color='b', linewidth=2, fill=False))
    plt.show()


def find_cells(stack):
    """Label spots"""
    assert psutil.virtual_memory()[1] > 8*stack.nbytes, \
        'Less than 1GiB of memory available'
    threshold = stack.max() / 5
    thresholded_stack = stack > threshold
    min_idxs = [min(indices) for indices in np.where(thresholded_stack)]
    max_idxs = [max(indices) for indices in np.where(thresholded_stack)]
    thresholded_stack = thresholded_stack[min_idxs[0]:max_idxs[0], min_idxs[1]:
                                          max_idxs[1], min_idxs[2]:max_idxs[2]]
    stack = stack[min_idxs[0]:max_idxs[0], min_idxs[1]:max_idxs[1], min_idxs[
        2]:max_idxs[2]]
    labels = measure.label(thresholded_stack)
    cells = pd.DataFrame()
    for label in np.unique(labels):
        locations = np.where(labels == label)
        cells.loc[label, 'X'] = min_idxs[2] + np.mean(locations[2])
        cells.loc[label, 'Y'] = min_idxs[1] + np.mean(locations[1])
        cells.loc[label, 'Z'] = min_idxs[0] + np.mean(locations[0])
        cells.loc[label, 'Mean intensity'] = np.mean(stack[labels == label])
        cells.loc[label, 'Max. intensity'] = np.max(stack[labels == label])
        cells.loc[label, 'Volume'] = np.sum(labels == label)
    return cells[1:]


if __name__ == "__main__":
    # # Open & display stacks in RGB
    # stack1 = (np.random.rand(50, 200, 150)*255).astype(np.uint8)
    # stack2 = (np.random.rand(50, 200, 150)*255).astype(np.uint8)
    # rgb_stack = stacks2rgb(stack1, stack2)
    # plot_stack(rgb_stack)

    # stack1 = read_stack('Examples/SPIM_example.tif')
    # stack2 = read_stack('Examples/SPIM_example2.tif')
    # stack3 = read_stack('Examples/SPIM_example3.tif')
    # stack3 = adjust_gamma(stack3, 0.5)
    # rgb_stack = stacks2rgb(stack1, stack2, stack3)
    # plot_stack(rgb_stack)

    # # Find some cells
    # stack = np.zeros((50, 200, 150))
    # points = (stack.shape*np.random.rand(8, 3)).T.astype(np.int)
    # stack[[point for point in points]] = 1
    # stack = filters.gaussian_filter(stack, 1)

    # stack = read_stack('Examples/SPIM_example.tif')
    stack = read_stack('Examples/SPIM_example.tif')[160:190, 100:450, 300:600]

    cells = find_cells(stack)
    print(cells)
    plot_stack(stack, cells)

    # # Aggregate excel files from SPIM analysis with MATLAB
    # DCs = pd.read_excel(
    #     '../Data/SPIM_OutputExample/20k-1_BMDC_coordinates.xls',
    #     sheetname='Position', skiprows=1)[[
    #         'Position X', 'Position Y', 'Position Z']]
    # DCs.columns = ['X', 'Y', 'Z']
    # DCs['Distance to Closest HEV'] = pd.read_excel(
    #     '../Data/SPIM_OutputExample/20k-1_DC-HEV.xls',
    #     sheetname='Distances_to_closest_HEV', header=None)
    # DCs['Distance to LN Center'] = pd.read_excel(
    #     '../Data/SPIM_OutputExample/20k-1_DC-HEV.xls',
    #     sheetname='Distances_to_Center_of_LN', header=None)
    # DCs['Cell Type'] = 'Dendritic Cell'
    #
    # T_cells = pd.read_excel(
    #     '../Data/SPIM_OutputExample/20k-1_OT-I_coordinates.xls',
    #     sheetname='Position', skiprows=1)[[
    #         'Position X', 'Position Y', 'Position Z']]
    # T_cells.columns = ['X', 'Y', 'Z']
    # T_cells['Distance to Closest HEV'] = pd.read_excel(
    #     '../Data/SPIM_OutputExample/20k-1_T-HEV.xls',
    #     sheetname='Distances_to_closest_HEV', header=None)
    # T_cells['Distance to LN Center'] = pd.read_excel(
    #     '../Data/SPIM_OutputExample/20k-1_T-HEV.xls',
    #     sheetname='Distances_to_Center_of_LN', header=None)
    # T_cells['Distance to Closest DC'] = pd.read_excel(
    #     '../Data/SPIM_OutputExample/20k-1_T-DC.xls',
    #     sheetname='Distances_between_cell_types', header=None)
    # T_cells['Cell Type'] = 'T Cell'
    #
    # positions = DCs.append(T_cells)
    # print(positions)
    #
    # df = positions[[
    #     'Cell Type', 'Distance to Closest HEV', 'Distance to LN Center']]
    # df.set_index([df.index, 'Cell Type']).unstack('Cell Type').swaplevel(
    #     0, 1, axis=1).sort_values(axis=1)
    # print(df)
