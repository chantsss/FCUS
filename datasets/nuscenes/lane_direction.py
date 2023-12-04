import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from shapely import affinity
from shapely.geometry import MultiPolygon


def get_lane_dir(centerline_array):
    direction = list()
    for i in range(len(centerline_array)):
        last_i, next_i = i - 1, i + 1
        if i == 0:
            last_i = i
        elif i == len(centerline_array) - 1:
            next_i = i
        transition = centerline_array[next_i][:2] - centerline_array[last_i][:2]
        direction.append(transition / np.linalg.norm(transition))
    direction = np.stack(direction)
    return direction


def get_lane_mask(nusc_map, lane_token, local_box, canvas_size):
    polygon_token = nusc_map.get('lane', lane_token)['polygon_token']
    polygon = nusc_map.extract_polygon(polygon_token)
    lane_mask = nusc_map.explorer._polygon_geom_to_mask([polygon], local_box, 'lane', canvas_size)
    return lane_mask


def get_dir_mask(cline_m, cline_dir, mask):
    x, y = np.where(mask == 1)
    if len(x) == 0:
        return np.zeros([*mask.shape, 2])
    else:
        x_size = x.max() - x.min() + 1
        y_size = y.max() - y.min() + 1

    with cp.cuda.Device(0):
        cline_m, cline_dir, mask = cp.asarray(cline_m), cp.asarray(cline_dir), cp.asarray(mask),
        cline_m = cline_m - cp.asarray([y.min(), x.min()])
        mesh = [dim[..., None] for dim in cp.meshgrid(*[cp.asarray(range(dim)) for dim in [x_size, y_size]][::-1])]
        rel_pos = cp.stack([mesh[0] - cline_m[:, 0], mesh[1] - cline_m[:, 1]], axis=-1)
        distance = cp.linalg.norm(rel_pos, axis=-1)
        grid_label = cp.argmin(distance, axis=-1)[..., None]
        dir_mask_patch = (grid_label == cp.arange(len(cline_m)))[..., None] * cline_dir
        dir_mask_patch = cp.sum(dir_mask_patch, axis=-2) * mask[x.min():x.max()+1, y.min():y.max()+1, None]
        dir_mask = cp.zeros([*mask.shape, 2])
        dir_mask[x.min():x.max()+1, y.min():y.max()+1] = dir_mask_patch
        dir_mask_out = cp.asnumpy(dir_mask)

    return dir_mask_out


def get_lane_polygon(nusc_map, patch_box):
    """
     Retrieve the polygons of lanes within the specified patch.
     :param patch_box: Patch box defined as [x_center, y_center, height, width].
     :param layer_name: name of map layer to be extracted.
     :return: List of Polygon in a patch box.
     """

    patch_x = patch_box[0]
    patch_y = patch_box[1]

    patch = nusc_map.explorer.get_patch_coord(patch_box, 0)

    records = getattr(nusc_map, 'lane')

    polygon_list = []

    for record in records:
        polygon = nusc_map.extract_polygon(record['polygon_token'])

        if polygon.is_valid:
            new_polygon = polygon.intersection(patch)
            if not new_polygon.is_empty:
                new_polygon = affinity.rotate(new_polygon, 0,
                                              origin=(patch_x, patch_y), use_radians=False)
                new_polygon = affinity.affine_transform(new_polygon,
                                                        [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                if new_polygon.geom_type is 'Polygon':
                    new_polygon = MultiPolygon([new_polygon])
                polygon_list.append((record['token'], new_polygon))

    return polygon_list


def get_direction_mask(nusc_map, patch_box, canvas_size):
    """
    Return list of map mask layers of the specified patch.
    :param patch_box: Patch box defined as [x_center, y_center, height, width]. If None, this plots the entire map.
    :param patch_angle: Patch orientation in degrees. North-facing corresponds to 0.
    :param layer_names: A list of layer names to be extracted, or None for all non-geometric layers.
    :param canvas_size: Size of the output mask (h, w). If None, we use the default resolution of 10px/m.
    :return: Stacked numpy array of size [c x h x w] with c channels and the same width/height as the canvas.
    """

    # Get geometry of each layer.
    polygon_list = get_lane_polygon(nusc_map, patch_box)

    # Convert geometry of each layer into mask and stack them into a numpy tensor.
    # Convert the patch box from global coordinates to local coordinates by setting the center to (0, 0).
    local_box = (0.0, 0.0, patch_box[2], patch_box[3])

    dir_mask_list = list()
    for lane_token, polygon in polygon_list:
        mask = nusc_map.explorer._polygon_geom_to_mask([polygon], local_box, 'lane', canvas_size)
        centerline = np.array(nusc_map.discretize_lanes([lane_token], 1)[lane_token])
        if len(centerline) == 0:  # bad data
            print('missed centerline')
            continue
        centerline_m = (centerline[:, :2] - (np.array(patch_box[:2]) - np.array(patch_box[2:][::-1]) / 2)) * np.array([3, 3])
        centerline_dir = get_lane_dir(centerline)
        dir_mask = get_dir_mask(centerline_m, centerline_dir, mask)
        dir_mask_list.append(dir_mask)
    if len(dir_mask_list) != 0:
        dir_mask_total = np.sum(np.stack(dir_mask_list), axis=0)
    else:
        dir_mask_total = np.zeros(2*patch_box[2], 2*patch_box[2], 2)

    return dir_mask_total


def visualization_lane_direction(direction_mask):
    norm_x = np.round(np.linalg.norm(direction_mask, axis=-1))
    plt.imshow(norm_x, origin='lower')
    a, b = np.where(norm_x > 0)
    select_idx = np.random.randint(0, len(a), 20)
    for idx in select_idx:
        plt.arrow(b[idx], a[idx], direction_mask[a[idx], b[idx]][0] * 50, direction_mask[a[idx], b[idx]][1] * 50,
                  color='r', head_width=20)
    plt.show()
