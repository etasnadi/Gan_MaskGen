import cellpose.dynamics as dyn
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage
import tifffile
from skimage import measure
from skimage.morphology import skeletonize, dilation, erosion

from common import quantize, dequantize


def get_skeleton(img_in, dilate=True):
    # binarize
    bin_im = np.zeros((img_in.shape[0], img_in.shape[1]))
    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            # if img_in[i, j, :].max() > 0.5:
            if len(img_in.shape) > 2:
                if img_in[i, j, :].max() > 60:
                    bin_im[i, j] = 1.
            else:
                if img_in[i, j] > 0:
                    bin_im[i, j] = 1.

    image = bin_im.copy()

    # perform skeletonization
    skeleton = skeletonize(image)
    if dilate:
        skeleton = dilation(skeleton)

    return skeleton


def _extend_ske(T, y, x, y_ske, x_ske, Lx, niter):
    """ run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)

    Parameters
    --------------

    T: float64, array
        _ x Lx array that diffusion is run in

    y: int32, array
        pixels in y inside mask

    x: int32, array
        pixels in x inside mask

    ymed: int32
        center of mask in y

    xmed: int32
        center of mask in x

    Lx: int32
        size of x-dimension of masks

    niter: int32
        number of iterations to run diffusion

    Returns
    ---------------

    T: float64, array
        amount of diffused particles at each pixel

    """
    for t in range(niter):
        T[y_ske * Lx + x_ske] += 1.
        T[y_ske * Lx + x_ske] = T[y_ske * Lx + x_ske].max()
        T[y * Lx + x] = 1 / 9. * (T[y * Lx + x] + T[(y - 1) * Lx + x] + T[(y + 1) * Lx + x] +
                                  T[y * Lx + x - 1] + T[y * Lx + x + 1] +
                                  T[(y - 1) * Lx + x - 1] + T[(y - 1) * Lx + x + 1] +
                                  T[(y + 1) * Lx + x - 1] + T[(y + 1) * Lx + x + 1])

    T[y_ske * Lx + x_ske] = T[y_ske * Lx + x_ske].max()
    # T[y_ske * Lx + x_ske] = np.quantile(T[y_ske * Lx + x_ske], .8)
    return T


def masks_to_flows_cpu_ske_heat(masks, device=None):
    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    dist_m = np.zeros((Ly, Lx), np.float64)

    slices = scipy.ndimage.find_objects(masks)
    for i, si in enumerate(slices):
        if si is not None:
            mask = masks[si]
            mask = mask * (mask == i + 1)
            ske = get_skeleton(mask, True)
            # ske = get_skeleton(mask, False)
            sr, sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            y, x = np.nonzero(mask == (i + 1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1

            y_ske, x_ske = np.nonzero(ske == 1)
            y_ske = y_ske.astype(np.int32) + 1
            x_ske = x_ske.astype(np.int32) + 1

            niter = 2 * np.int32(np.ptp(x) + np.ptp(y))
            # niter = 15
            T = np.zeros((ly + 2) * (lx + 2), np.float64)
            T = _extend_ske(T, y, x, y_ske, x_ske, np.int32(lx), niter)
            # tmax = np.max(T)
            # T[(y_ske + 1) * lx + x_ske + 1] = tmax
            # T[(y + 1) * lx + x + 1] = np.log(1. + T[(y + 1) * lx + x + 1])

            dy = T[(y + 1) * lx + x] - T[(y - 1) * lx + x]
            dx = T[y * lx + x + 1] - T[y * lx + x - 1]
            mu[:, sr.start + y - 1, sc.start + x - 1] = np.stack((dy, dx))
            dist_m[sr, sc] = T[:lx * ly].reshape(ly, lx)[1:, 1:]

    mu /= (1e-20 + (mu ** 2).sum(axis=0) ** 0.5)
    return mu, dist_m


def masks_to_flows_cpu_no_grad(masks, device=None):
    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    dist_m = np.zeros((Ly, Lx), np.float64)

    slices = scipy.ndimage.find_objects(masks)
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            y, x = np.where(masks[sr, sc] == (i + 1))

            mask_slice_o = np.zeros(masks[sr, sc].shape)
            mask_slice_o[y, x] = masks[sr, sc][y, x]
            mask_slice_o = mask_slice_o / mask_slice_o.max()
            mask_slice = dilation(np.pad(mask_slice_o, 2))

            skele_slice = get_skeleton(mask_slice_o, False)
            # skele_slice = get_skeleton(mask_slice_o, True)
            skele_slice = np.pad(skele_slice, 2)

            res = np.array(skele_slice.copy())
            _next = np.array(skele_slice.copy())
            current = np.array(skele_slice.copy())
            j = 1
            while True:
                _next = (dilation(_next) * mask_slice)
                if (current == _next).all() and j != 1:
                    break
                res = res + _next
                current = _next
                j += 1

            points = np.nonzero(res)
            res_nz = res[points]
            res[points] = res_nz - res_nz.max()
            res = np.abs(res)

            we = np.tanh(np.linspace(0, 1, int(np.max(res))))
            we = we / we.max()

            resw = np.zeros(res.shape)
            for x, y in zip(*np.nonzero(res)):
                _x, _y = int(x), int(y)
                resw[_x, _y] = we[int(res[_x, _y] - 1)]

            x, y = np.nonzero(res * np.pad(mask_slice_o, 2))
            points_ske_x, points_ske_y = np.nonzero(skele_slice * np.pad(mask_slice_o, 2))

            dx, dy = np.zeros(res.shape), np.zeros(res.shape)
            for _x, _y in zip(x, y):
                x_dif = _x - points_ske_x
                y_dif = _y - points_ske_y

                xs = np.sign(y_dif[np.argmin(np.abs(x_dif))]) * -1
                ys = np.sign(x_dif[np.argmin(np.abs(y_dif))]) * -1

                xs = xs if xs != 0 else 1
                ys = ys if ys != 0 else 1

                dx[_x, _y] = we[int(res[_x, _y]) - 1] * ys
                dy[_x, _y] = we[int(res[_x, _y]) - 1] * xs

            dx = dx / np.abs(dx.max())
            dy = dy / np.abs(dy.max())
            dx = dx[2:-2, 2:-2] + mu[0, si[0].start:si[0].stop, si[1].start:si[1].stop]
            dy = dy[2:-2, 2:-2] + mu[1, si[0].start:si[0].stop, si[1].start:si[1].stop]
            _dist = (resw[2:-2, 2:-2] * mask_slice_o) + dist_m[si[0].start:si[0].stop, si[1].start:si[1].stop]

            mu[0, si[0].start:si[0].stop, si[1].start:si[1].stop] = dx
            mu[1, si[0].start:si[0].stop, si[1].start:si[1].stop] = dy

            dist_m[si[0].start:si[0].stop, si[1].start:si[1].stop] = _dist

    return mu, dist_m


def masks_to_flows(masks, method='cp'):
    if masks.ndim == 2:
        if method == 'ske_dist':
            mu, dist_m = masks_to_flows_cpu_no_grad(masks)
        elif method == 'ske_heat':
            mu, dist_m = masks_to_flows_cpu_ske_heat(masks)
        elif method == 'cp':
            mu, dist_m = dyn.masks_to_flows_cpu(masks)
        else:
            raise ValueError('Invalid method use one of ske_dist, ske_heat, cp')
        return mu, dist_m
    else:
        raise ValueError('masks_to_flows only takes 2D or 3D arrays')


def labels_to_flows(labels, files=None, method='cp'):
    nimg = len(labels)
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis, :, :] for n in range(nimg)]

    if labels[0].shape[0] == 1 or labels[0].ndim < 3:
        veci = [masks_to_flows(labels[n][0], method)[0] for n in range(nimg)]
        # concatenate flows with cell probability
        flows = []
        for n in range(nimg):
            flows.append(np.concatenate((labels[n][[0]], labels[n][[0]] > 0.5, veci[n]), axis=0).astype(np.float32))

        if files is not None:
            for flow, file in zip(flows, files):
                file_name = os.path.splitext(file)[0]
                tifffile.imsave(file_name + '_flows.tif', flow)
    else:
        flows = [labels[n].astype(np.float32) for n in range(nimg)]
    return flows


def encode_labels(im, method='cp'):
    flow = labels_to_flows([im], method=method)
    cellprob = flow[0][1, ...]

    dP = flow[0][2:, ...]
    dPq = quantize(dP.copy() * 3)
    cellprobq = (cellprob > 0).astype(np.uint8) * 255
    cellprobq = np.expand_dims(cellprobq, 0)

    repr_ = np.concatenate([cellprobq, dPq], 0)

    repr_ = repr_.transpose((1, 2, 0))
    return repr_, dP


def get_masks(p, iscell=None, rpad=20, flows=None, threshold=0.4):
    """ create masks using pixel convergence after running dynamics

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.

    Parameters
    ----------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].

    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are
        iscell False to stay in their original location.

    rpad: int (optional, default 20)
        histogram edge padding

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded
        (if flows is not None)

    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using
        `remove_bad_flow_masks`.

    Returns
    ---------------

    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims == 3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               np.arange(shape0[2]), indexing='ij')
        elif dims == 2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5 - rpad, shape0[i] + .5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = dyn.maximum_filter1d(hmax, 5, axis=i)

    base_seeds = np.where(h > 1)
    components = []
    for s in zip(*base_seeds):
        close_comps = []
        for i, comp in enumerate(components):
            for cb in comp:
                if np.abs(cb[0] - s[0]) <= 2 and np.abs(cb[1] - s[1]) <= 2:
                    close_comps.append(i)
                    break
        if len(close_comps) == 1:
            components[close_comps[0]].append(s)
        elif len(close_comps) > 1:
            # print('Multiple close comp')
            components[np.argmax([components[ic] for ic in close_comps])].append(s)
            # for ic in close_comps:
            #     components[ic].append(s)
        else:
            components.append([s])

    comps = np.zeros(h.shape)
    for i, c in enumerate(components):
        comps[list(zip(*c))] = i + 1

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

    ax[0][0].imshow(np.pad(flows[0], 20))
    ax[0][1].imshow(np.pad(flows[1], 20))

    ax[1][0].imshow(np.log(h + 1))
    ax[1][1].imshow(comps)

    fig.tight_layout()
    # plt.show()

    # merge conponents
    # components_m = components
    # components_m_next = []
    # merged = set()
    #
    # while len(components_m) != len(components_m_next):
    #     components_m = components_m_next if components_m_next else components_m
    #     components_m_next = []
    #
    #     for i, c in enumerate(components_m):
    #         # if i in merged:
    #         #     continue
    #
    #         nc = c
    #         sc = set(c)
    #         for j, c2 in enumerate(components_m[i+1:]):
    #             if len(sc.intersection(c2)) > int(min(len(c), len(c2)) * 0.8):
    #                 nc = list(sc.union(c2))
    #                 merged.add(i+j+1)
    #         components_m_next.append(nc)
    #
    # # check
    # for i, c in enumerate(components_m[:-1]):
    #     sc = set(c)
    #     for c2 in components_m[i+1 :]:
    #         if len(sc.intersection(c2)) > 0:
    #             print('Overlaping components')
    #             print(len(c), c)
    #             print(len(c2), c2)
    #             print(len(sc.intersection(c2)), sc.intersection(c2))

    components = [c for c in components if h[list(zip(*c))].max() > 10]

    comps = np.zeros(h.shape, np.int32)
    for i, c in enumerate(components):
        comps[list(zip(*c))] = i + 1

    M = comps.copy()
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    _, counts = np.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    for i in np.nonzero(counts > big)[0]:
        M0[M0 == i] = 0
    _, M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    if M0.max() > 0 and threshold is not None and threshold > 0 and flows is not None:
        # M0 = remove_bad_flow_masks(M0, flows, threshold=threshold, use_gpu=use_gpu, device=device)
        _, M0 = np.unique(M0, return_inverse=True)
        M0 = np.reshape(M0, shape0).astype(np.int32)

    return M0


def get_masks2(p, iscell=None, rpad=20, flows=None, threshold=0.4):
    """ create masks using pixel convergence after running dynamics

    Makes a histogram of final pixel locations p, initializes masks
    at peaks of histogram and extends the masks from the peaks so that
    they include all pixels with more than 2 final pixels p. Discards
    masks with flow errors greater than the threshold.

    Parameters
    ----------------

    p: float32, 3D or 4D array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx].

    iscell: bool, 2D or 3D array
        if iscell is not None, set pixels that are
        iscell False to stay in their original location.

    rpad: int (optional, default 20)
        histogram edge padding

    threshold: float (optional, default 0.4)
        masks with flow error greater than threshold are discarded
        (if flows is not None)

    flows: float, 3D or 4D array (optional, default None)
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. If flows
        is not None, then masks with inconsistent flows are removed using
        `remove_bad_flow_masks`.

    Returns
    ---------------

    M0: int, 2D or 3D array
        masks with inconsistent flow masks removed,
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)
    if iscell is not None:
        if dims == 3:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               np.arange(shape0[2]), indexing='ij')
        elif dims == 2:
            inds = np.meshgrid(np.arange(shape0[0]), np.arange(shape0[1]),
                               indexing='ij')
        for i in range(dims):
            p[i, ~iscell] = inds[i][~iscell]

    for i in range(dims):
        pflows.append(p[i].flatten().astype('int32'))
        edges.append(np.arange(-.5 - rpad, shape0[i] + .5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = dyn.maximum_filter1d(hmax, 5, axis=i)

    ht = np.zeros(h.shape)
    ht[np.where(h > 1)] = 1
    htt = erosion(erosion(dilation(dilation(ht))))
    comps = measure.label(htt)

    M = comps.copy()
    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    _, counts = np.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.4
    for i in np.nonzero(counts > big)[0]:
        M0[M0 == i] = 0
    _, M0 = np.unique(M0, return_inverse=True)
    M0 = np.reshape(M0, shape0)

    if M0.max() > 0 and threshold is not None and threshold > 0 and flows is not None:
        # M0 = dyn.remove_bad_flow_masks(M0, flows, threshold=15, use_gpu=False, device=None)
        _, M0 = np.unique(M0, return_inverse=True)
        M0 = np.reshape(M0, shape0).astype(np.int32)

    return M0


def reconstruct_mask(im_repr, cellprob_threshold=0., flow_threshold=.4, niter=200,
                     interp=False, use_gpu=False, device=None):
    repr_ = im_repr.transpose((2, 0, 1))

    cellprob_quant = repr_[:1, ...]
    dP_quant = repr_[1:, ...]

    cellprob = cellprob_quant.astype(np.float32)
    dP_dequant = dequantize(dP_quant)

    dP = dP_dequant
    iscell = (cellprob > cellprob_threshold).squeeze().astype(np.bool)

    # print('dP shape:', dP.shape, cellprob.shape)

    p = dyn.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5.,
                         niter=niter, interp=interp, use_gpu=use_gpu,
                         device=device)

    maski = get_masks2(p, iscell=iscell,
                       flows=dP, threshold=flow_threshold)

    maski = erosion(dilation(maski)) * iscell

    # maski = dyn.get_masks(p, iscell=(cellprob > cellprob_threshold),
    #                       flows=dP, threshold=flow_threshold,
    #                       use_gpu=use_gpu, device=device)

    return maski.astype(np.uint8), p
