import matplotlib.cm
import matplotlib.colors

import numpy as np


# End-point-error visualization as described in "Object Scene Flow" by Menze et
# al. (2018), based on implementation by Mehl at https://github.com/cv-stuttgart/flow_library.
def end_point_error_abs(uv, uv_target, mask=None, mask_color=(0, 0, 0, 1), nan_color=(0, 0, 0, 1)):
    # Colors as defined by Mehl (they differ slightly from the "Object Scene
    # Flow" paper), but represent a logarithmic scale (i.e. the numbers in the
    # paper are rounded).
    colors = [
        (0.1875, [49, 53, 148]),
        (0.375, [69, 116, 180]),
        (0.75, [115, 173, 209]),
        (1.5, [171, 216, 233]),
        (3, [223, 242, 248]),
        (6, [254, 223, 144]),
        (12, [253, 173, 96]),
        (24, [243, 108, 67]),
        (48, [215, 48, 38]),
        (np.inf, [165, 0, 38])
    ]

    # convert to RGBA in range [0, 1]
    colors = [(th, [r / 255.0, g / 255.0, b / 255.0, 1.0]) for (th, (r, g, b)) in colors]

    # compute end-point-error
    epe = np.linalg.norm(uv_target - uv, axis=-1, ord=2)

    # get NaN/inifinite values
    nan = ~np.isfinite(epe)

    # reset NaN values so we don't run into any problems while coloring
    epe = np.nan_to_num(epe)

    # initialize output to opaque black
    rgba = np.zeros((*epe.shape[:2], 4))
    rgba[:, :, 3] = 1.0

    # assign colors based on thresholds
    for th, color in reversed(colors):
        rgba[epe < th, :] = color

    # set NaN/infinite and masked values accordingly
    rgba[nan, :] = np.array(nan_color)

    if mask is not None:
        rgba[~mask, :] = np.array(mask_color)

    return rgba


def end_point_error(uv, uv_target, mask=None, ord=2, cmap='gray', vmin=0.0, vmax=None, mask_color=(0, 0, 0, 1)):
    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    d = np.linalg.norm(uv_target - uv, axis=-1, ord=ord)

    if mask is not None:
        d = d * mask

    rgb = cmap(norm(d))

    if mask is not None:
        rgb[~mask] = np.asarray(mask_color)

    return rgb
