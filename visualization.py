import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils.dataset import RSRD  # Ensure this module is available
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import os
import fnmatch
from PIL import Image

def find_files_with_pattern(directory, pattern):
    matched_files = []
    for root, dirs, files in os.walk(directory):
        for ext in ['*.png', '*.jpg']:
            for filename in fnmatch.filter(files, f"*{pattern}*{ext}"):
                matched_files.append(os.path.join(root, filename))
    return matched_files

def plot_bev_mesh(rsrd, path_pred, save_path):
    x = -torch.arange(rsrd.num_grids_x) * rsrd.grid_res[0] + rsrd.roi_x[1] - rsrd.grid_res[0]/2
    z = torch.arange(rsrd.num_grids_z) * rsrd.grid_res[2] + rsrd.roi_z[0] + rsrd.grid_res[2]/2
    Z, X = np.meshgrid(np.array(z), np.array(x))

    with open(path_pred, 'rb') as f:
        ele_pred = pickle.load(f)
    ele_pred = np.array(ele_pred)
    ele_pred = np.transpose(np.flip(ele_pred, axis=0))
    ele_max = np.max(ele_pred)
    ele_min = np.min(ele_pred)

    fig = plt.figure(figsize=(10, 5), dpi=250)
    ax1 = fig.add_subplot(111)
    im1 = ax1.pcolormesh(Z, X, ele_pred, vmax=ele_max, vmin=ele_min, cmap='plasma')
    plt.colorbar(im1, ax=ax1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close(fig)

def plot_3d_surface(rsrd, path_pred, save_path):
    x = torch.arange(rsrd.num_grids_x) * rsrd.grid_res[0] + rsrd.roi_x[0] - rsrd.grid_res[0]/2
    z = -torch.arange(rsrd.num_grids_z) * rsrd.grid_res[2] + rsrd.roi_z[1] + rsrd.grid_res[2]/2
    X, Z = np.meshgrid(np.array(x), np.array(z))

    with open(path_pred, 'rb') as f:
        ele_pred = pickle.load(f)
    ele_pred = np.array(ele_pred)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Z, ele_pred, cmap=cm.coolwarm, rstride=1, cstride=1, linewidth=1, antialiased=False)

    # Customizing the axes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_box_aspect([rsrd.num_grids_x, rsrd.num_grids_z, 10])
    ax.set_zlim(-20, 20)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    plt.savefig(save_path, format='png')
    plt.close(fig)

if __name__ == '__main__':
    # This script visualizes the elevation inference. Before that, save the ele_pred as .pkl file
    rsrd = RSRD()

    directory = './bev_pred'
    files = os.listdir(directory)

    thumbnail_size=(500, 500)

    rows = 3
    cols = 3

    total_width = cols * thumbnail_size[0]
    total_height = rows * thumbnail_size[1]

    for index, path_pred in enumerate(files):
        print(f"{index}/{len(files)}")
        # Save the plots to files
        item_name = path_pred.split(".pkl")[0] 
        new_item_save_path = f'./visualizations/{item_name}'
        os.makedirs(new_item_save_path, exist_ok=True) 
        plot_bev_mesh(rsrd, f'{directory}/{path_pred}', f'{new_item_save_path}/bev_mesh.png')
        plot_3d_surface(rsrd, f'{directory}/{path_pred}', f'{new_item_save_path}/3d_surface.png')

        directory_search = '../RSRD-dense'  # Current directory
        pattern = item_name

        matched_files = find_files_with_pattern(directory_search, pattern)
        merged_image = Image.new('RGB', (total_width, total_height))

        for index, image_path in enumerate(matched_files):
            img = Image.open(image_path)
            img.thumbnail(thumbnail_size)
            
            # Calculate the position of the current image
            x = (index % cols) * thumbnail_size[0]
            y = (index // cols) * thumbnail_size[1]
            
            # Paste the image onto the canvas
            merged_image.paste(img, (x, y))

        # Save the merged image
        merged_image.save(f'{new_item_save_path}/context.png')

