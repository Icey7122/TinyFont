# -*- coding: utf-8 -*-
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Check for optional skimage dependency used for skeletonization.
try:
    from skimage.morphology import skeletonize
    from skimage.measure import label, regionprops
    SKIMAGE_AVAIL = True
except ImportError:
    SKIMAGE_AVAIL = False

# -----------------------------------------------------------------------------
# Basic geometric functions
# -----------------------------------------------------------------------------

def distance(p0, p1):
    """Compute Euclidean distance between two 2D points.

    Args:
        p0: Tuple[float, float], first point (x, y).
        p1: Tuple[float, float], second point (x, y).

    Returns:
        float: Euclidean distance between ``p0`` and ``p1``.
    """

    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

def pt2seg(p, seg):
    (x,y) = p
    ((x0,y0),(x1,y1)) = seg
    l2 = (x0-x1)**2 + (y0-y1)**2
    if l2 == 0: return (x0,y0), distance(p,seg[0]), 0
    """Project a point onto a line segment and compute distance.

    Args:
        p: Tuple[float, float], point to project.
        seg: Tuple[Tuple[float, float], Tuple[float, float]], segment endpoints.

    Returns:
        Tuple[Tuple[float, float], float, int]: Projection point, distance from
        ``p`` to the projection, and a placeholder integer (always 0).
    """

    t = ((x - x0) * (x1 - x0) + (y - y0) * (y1 - y0)) / l2
    t = max(0, min(1, t))
    proj = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
    d = distance(p, proj)
    return proj, d, 0

# -----------------------------------------------------------------------------
# Core stroke extraction algorithm
# -----------------------------------------------------------------------------

def raster_to_strokes(mtx, strw=10, simplify_eps=None, spur_len=None, join_dist=None):
    """Extract stroke polylines from a raster matrix via skeletonization.

    This function converts a sparse matrix-like mapping ``mtx`` into a
    binary image, computes a morphological skeleton, identifies branch
    segments and junctions, prunes short spurs, merges colinear/adjacent
    segments and returns a list of simplified strokes.

    Args:
        mtx: dict-like mapping {(x,y): value, ...} with ``'size'`` key for
            image width and height. Nonzero values are treated as ink.
        strw: int, nominal stroke width used to set pruning thresholds.
        simplify_eps: float or None, Ramer–Douglas–Peucker epsilon for
            polyline simplification.
        spur_len: float or None, explicit length threshold for spur pruning.
        join_dist: float or None, maximum distance to join nearby endpoints.

    Returns:
        list: List of strokes, each stroke is a list of (x, y) points.

    Raises:
        RuntimeError: If scikit-image is not available.
    """

    if not SKIMAGE_AVAIL:
        raise RuntimeError("scikit-image is required: pip install scikit-image")

    w, h = mtx['size']

    # 1. Rasterize mtx into a boolean image.
    img = np.zeros((h, w), dtype=bool)
    for key, v in mtx.items():
        if key == 'size':
            continue
        if not (isinstance(key, tuple) and len(key) == 2):
            continue
        x, y = key
        if isinstance(x, int) and isinstance(y, int):
            if 0 <= x < w and 0 <= y < h:
                img[y, x] = bool(v)

    if not img.any():
        return []

    # 2. Skeletonize binary image.
    skel = skeletonize(img)

    # 3. Compute degree (number of neighbors) for skeleton pixels.
    def get_neighbors(x, y):
        nbs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                xx, yy = x + dx, y + dy
                if 0 <= xx < w and 0 <= yy < h and skel[yy, xx]:
                    nbs.append((xx, yy))
        return nbs

    deg_map = np.zeros((h, w), dtype=int)
    for yy in range(h):
        for xx in range(w):
            if skel[yy, xx]:
                deg_map[yy, xx] = len(get_neighbors(xx, yy))

    # 4. Identify junction regions where degree > 2 and compute centroids.
    junc_mask = (deg_map > 2)
    junc_labels, num_juncs = label(junc_mask, return_num=True, connectivity=2)
    junc_props = regionprops(junc_labels)

    junc_centroids = {}
    for prop in junc_props:
        cy, cx = prop.centroid
        junc_centroids[prop.label] = (cx, cy)

    # 5. Extract branch components (skeleton pixels not part of junctions).
    branch_mask = skel & (~junc_mask)
    branch_labels, num_branches = label(branch_mask, return_num=True, connectivity=2)

    raw_strokes = []

    def find_connected_junction(px, py):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = int(px) + dx, int(py) + dy
                if 0 <= nx < w and 0 <= ny < h:
                    lid = junc_labels[ny, nx]
                    if lid > 0:
                        return junc_centroids[lid]
        return None

    for i in range(1, num_branches + 1):
        coords_y, coords_x = np.where(branch_labels == i)
        points = list(zip(coords_x, coords_y))
        if not points:
            continue

        # Order branch pixels into a path by walking from an endpoint.
        if len(points) == 1:
            ordered_path = points
        else:
            local_set = set(points)
            endpoints = []
            for px, py in points:
                n_count = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        if (px + dx, py + dy) in local_set:
                            n_count += 1
                if n_count <= 1:
                    endpoints.append((px, py))

            start_node = endpoints[0] if endpoints else points[0]
            ordered_path = [start_node]
            visited = {start_node}
            curr = start_node

            while len(ordered_path) < len(points):
                found = False
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nb = (curr[0] + dx, curr[1] + dy)
                        if nb in local_set and nb not in visited:
                            visited.add(nb)
                            ordered_path.append(nb)
                            curr = nb
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break

        head_junc = find_connected_junction(*ordered_path[0])
        if head_junc:
            ordered_path.insert(0, head_junc)
        if len(ordered_path) > 1 or not head_junc:
            tail_junc = find_connected_junction(*ordered_path[-1])
            if tail_junc:
                if not (head_junc and tail_junc and head_junc == tail_junc):
                    ordered_path.append(tail_junc)
        raw_strokes.append(ordered_path)

    # 5.5 Prune spurs shorter than spur_threshold.
    junc_coords_set = set(junc_centroids.values())
    spur_threshold = spur_len if spur_len is not None else max(4.0, float(strw) * 0.6)

    pruned_strokes = []
    for s in raw_strokes:
        if not s:
            continue
        if len(s) == 1:
            pruned_strokes.append(s)
            continue

        head_connected = tuple(s[0]) in junc_coords_set
        tail_connected = tuple(s[-1]) in junc_coords_set

        if not head_connected and not tail_connected:
            pruned_strokes.append(s)
        elif head_connected and tail_connected:
            pruned_strokes.append(s)
        else:
            length = sum(math.hypot(s[i][0] - s[i - 1][0], s[i][1] - s[i - 1][1]) for i in range(1, len(s)))
            if length > spur_threshold:
                pruned_strokes.append(s)

    def snap_endpoints(strokes, threshold=3.0):
        """Cluster and snap stroke endpoints within a distance threshold.

        Endpoints from all strokes that lie within ``threshold`` distance of each
        other are grouped and replaced by their centroid.

        Args:
            strokes: Sequence[Sequence[Tuple[float, float]]], list of strokes.
            threshold: float, maximum distance to consider endpoints the same.

        Returns:
            list: New strokes with endpoints snapped to cluster centroids.
        """
        if not strokes:
            return strokes

        # 1. Collect all endpoints (record stroke index and point index: 0=start, -1=end)
        endpoints = []
        for i, s in enumerate(strokes):
            if len(s) < 1: continue
            endpoints.append({'stroke_idx': i, 'point_idx': 0, 'coord': np.array(s[0], dtype=float)})
            if len(s) > 1:
                endpoints.append({'stroke_idx': i, 'point_idx': -1, 'coord': np.array(s[-1], dtype=float)})

        # 2. Clustering: flood-fill style grouping based on distance threshold.
        n = len(endpoints)
        clusters = []  # list of index lists representing endpoint clusters
        visited = [False] * n

        for i in range(n):
            if visited[i]: continue
            # start a new cluster
            current_cluster = [i]
            visited[i] = True
            
            # find points within `threshold` of any point in the current cluster
            # expand the cluster iteratively
            idx = 0
            while idx < len(current_cluster):
                ref_idx = current_cluster[idx]
                for j in range(n):
                    if not visited[j]:
                        dist = np.linalg.norm(endpoints[ref_idx]['coord'] - endpoints[j]['coord'])
                        if dist <= threshold:
                            visited[j] = True
                            current_cluster.append(j)
                idx += 1
            clusters.append(current_cluster)

        # 3. Compute centroids for each cluster and write them back to strokes
        new_strokes = [list(s) for s in strokes]
        for cluster in clusters:
            # compute centroid
            coords = [endpoints[idx]['coord'] for idx in cluster]
            centroid = tuple(np.mean(coords, axis=0))
            
            # update coordinates in the original strokes
            for idx in cluster:
                s_idx = endpoints[idx]['stroke_idx']
                p_idx = endpoints[idx]['point_idx']
                new_strokes[s_idx][p_idx] = centroid

        return new_strokes

    pruned_strokes = snap_endpoints(pruned_strokes, 5)

    # 7. Merge colinear/adjacent strokes into longer segments.
    def consolidate_strokes(strokes, angle_thresh=0.86):
        strokes = [list(s) for s in strokes]
        while True:
            changed = False
            adj_dict = {}
            for i, s in enumerate(strokes):
                if s is None:
                    continue
                p_start = (round(s[0][0], 1), round(s[0][1], 1))
                p_end = (round(s[-1][0], 1), round(s[-1][1], 1))
                adj_dict.setdefault(p_start, []).append((i, 0))
                if p_end != p_start:
                    adj_dict.setdefault(p_end, []).append((i, -1))

            for p, connections in adj_dict.items():
                if len(connections) < 2:
                    continue
                vecs = []
                for s_idx, end_type in connections:
                    s = strokes[s_idx]
                    k = min(len(s) - 1, 4)
                    if end_type == 0:
                        vx, vy = s[k][0] - s[0][0], s[k][1] - s[0][1]
                    else:
                        vx, vy = s[-1 - k][0] - s[-1][0], s[-1 - k][1] - s[-1][1]
                    norm = math.hypot(vx, vy)
                    vecs.append((vx / norm, vy / norm) if norm > 1e-3 else (0, 0))

                best_pair = None
                min_dot = -angle_thresh
                for ii in range(len(connections)):
                    for jj in range(ii + 1, len(connections)):
                        if connections[ii][0] == connections[jj][0]:
                            continue
                        dot = vecs[ii][0] * vecs[jj][0] + vecs[ii][1] * vecs[jj][1]
                        if dot < min_dot:
                            min_dot = dot
                            best_pair = (ii, jj)

                if best_pair:
                    idx1, idx2 = best_pair
                    s1_info, s2_info = connections[idx1], connections[idx2]
                    seg1, seg2 = strokes[s1_info[0]], strokes[s2_info[0]]
                    if s1_info[1] == 0:
                        seg1 = seg1[::-1]
                    if s2_info[1] == -1:
                        seg2 = seg2[::-1]
                    strokes[s1_info[0]] = seg1 + seg2[1:]
                    strokes[s2_info[0]] = None
                    changed = True
                    break
            strokes = [s for s in strokes if s is not None]
            if not changed:
                break
        return strokes

    final_strokes = consolidate_strokes(pruned_strokes)

    # 7.5 Simple topological connection by endpoint proximity.
    j_dist = join_dist if join_dist is not None else 2.5

    def simple_connect_strokes(strokes, d_lim):
        strokes = [list(s) for s in strokes]
        while True:
            merged = False
            for i in range(len(strokes)):
                if strokes[i] is None:
                    continue
                for j in range(i + 1, len(strokes)):
                    if strokes[j] is None:
                        continue
                    s1, s2 = strokes[i], strokes[j]
                    if (s1[-1][0] - s2[0][0]) ** 2 + (s1[-1][1] - s2[0][1]) ** 2 < d_lim ** 2:
                        strokes[i] = s1 + s2[1:]
                        strokes[j] = None
                        merged = True
                    elif (s1[-1][0] - s2[-1][0]) ** 2 + (s1[-1][1] - s2[-1][1]) ** 2 < d_lim ** 2:
                        strokes[i] = s1 + s2[::-1][1:]
                        strokes[j] = None
                        merged = True
                    elif (s1[0][0] - s2[0][0]) ** 2 + (s1[0][1] - s2[0][1]) ** 2 < d_lim ** 2:
                        strokes[i] = s1[::-1] + s2[1:]
                        strokes[j] = None
                        merged = True
                    elif (s1[0][0] - s2[-1][0]) ** 2 + (s1[0][1] - s2[-1][1]) ** 2 < d_lim ** 2:
                        strokes[i] = s2 + s1[1:]
                        strokes[j] = None
                        merged = True
                    if merged:
                        break
                if merged:
                    break
            if not merged:
                break
            strokes = [s for s in strokes if s is not None]
        return strokes

    final_strokes = simple_connect_strokes(final_strokes, j_dist)

    # 8. Filter extremely short segments and convert single points to short lines.
    def get_len(s):
        return sum(distance(s[i], s[i - 1]) for i in range(1, len(s)))

    cleaned_strokes = []
    for s in final_strokes:
        if len(s) == 1:
            # Convert isolated point to a minimal visible segment.
            x, y = s[0]
            cleaned_strokes.append([(x, y), (x + 0.1, y)])
        elif get_len(s) > 0.5:
            # Keep segments longer than a small threshold to suppress noise.
            cleaned_strokes.append(s)

    # Ramer–Douglas–Peucker simplification.
    def rdp(points, eps):
        if len(points) <= 2:
            return points
        p0, p1 = points[0], points[-1]

        def pld(p, a, b):
            if distance(a, b) < 1e-6:
                return distance(p, a)
            return pt2seg(p, (a, b))[1]

        max_d, idx = 0, -1
        for ii in range(1, len(points) - 1):
            d = pld(points[ii], p0, p1)
            if d > max_d:
                max_d, idx = d, ii
        if max_d <= eps:
            return [p0, p1]
        return rdp(points[: idx + 1], eps)[:-1] + rdp(points[idx:], eps)

    output = []
    for s in cleaned_strokes:
        closed = (len(s) > 2 and distance(s[0], s[-1]) < 1e-3)
        if closed:
            s = s[:-1]
        res = rdp(s, simplify_eps)
        if closed:
            res.append(res[0])
        if len(res) >= 2:
            output.append(res)

    def merge_close_points(strokes, dist_thresh=1.5):
        """Merge consecutive points inside strokes that are very close.

        For each stroke, consecutive points whose distance is less than
        ``dist_thresh`` are merged (the midpoint is used).

        Args:
            strokes: Sequence[Sequence[Tuple[float, float]]], list of strokes.
            dist_thresh: float, distance threshold for merging consecutive points.

        Returns:
            list: Strokes with near-duplicate consecutive points merged.
        """
        compacted_strokes = []
        for s in strokes:
            if len(s) < 2:
                compacted_strokes.append(s)
                continue
            
            new_s = [s[0]]
            for i in range(1, len(s)):
                p_prev = new_s[-1]
                p_curr = s[i]
                
                # Compute Euclidean distance between consecutive points.
                d = math.sqrt((p_prev[0] - p_curr[0])**2 + (p_prev[1] - p_curr[1])**2)
                
                if d < dist_thresh:
                    # Merge by replacing previous point with midpoint.
                    mid_p = ((p_prev[0] + p_curr[0]) / 2.0, (p_prev[1] + p_curr[1]) / 2.0)
                    new_s[-1] = mid_p
                else:
                    new_s.append(p_curr)
            
            if len(new_s) >= 2:
                compacted_strokes.append(new_s)
            else:
                # If a stroke collapses to a single point, add a tiny displacement
                # so the stroke still has two points (preserves structure).
                p = new_s[0]
                compacted_strokes.append([p, (p[0] + 0.1, p[1])])
        return compacted_strokes

    output = merge_close_points(output, 2)

    def get_stroke_length(s):
        """Return total polyline length of a stroke.

        Args:
            s: Sequence[Tuple[float, float]], stroke points.

        Returns:
            float: Sum of Euclidean distances between consecutive points.
        """

        return sum(distance(s[i], s[i - 1]) for i in range(1, len(s)))

    def cleanup_isolated_shards(strokes, length_thresh=2.5, dist_thresh=5.0):
        """Remove short strokes that are close to other strokes (noise).

        A stroke whose length is below ``length_thresh`` and whose representative
        point lies within ``dist_thresh`` distance of another stroke is treated
        as an isolated shard and removed.

        Args:
            strokes: Sequence[Sequence[Tuple[float, float]]], list of strokes.
            length_thresh: float, maximum length to consider a stroke a shard.
            dist_thresh: float, distance threshold for proximity to other strokes.

        Returns:
            list: Filtered strokes with isolated short shards removed.
        """
        if len(strokes) <= 1:
            return strokes

        keep_mask = [True] * len(strokes)

        # 1. Precompute stroke attributes to avoid repeated work.
        stroke_data = []
        for s in strokes:
            stroke_data.append({
                'points': s,
                'length': get_stroke_length(s),
                'is_short': get_stroke_length(s) < length_thresh,
            })

        # 2. For each short stroke, compute minimal distance to other strokes.
        for i in range(len(strokes)):
            if not stroke_data[i]['is_short']:
                continue

            min_d_to_others = float('inf')
            p_fragment = stroke_data[i]['points'][0]

            for j in range(len(strokes)):
                if i == j:
                    continue

                target_stroke = strokes[j]
                for k in range(len(target_stroke) - 1):
                    seg = (target_stroke[k], target_stroke[k + 1])
                    _, d, _ = pt2seg(p_fragment, seg)
                    if d < min_d_to_others:
                        min_d_to_others = d

                if min_d_to_others < dist_thresh:
                    break

            # 3. Mark as noise if shard is close to another stroke.
            if min_d_to_others < dist_thresh:
                keep_mask[i] = False

        return [strokes[i] for i, keep in enumerate(keep_mask) if keep]

    output = cleanup_isolated_shards(output, 8, 5)

    return output

# -----------------------------------------------------------------------------
# Compatibility helper functions
# -----------------------------------------------------------------------------

def im2mtx(im):
    """Convert a Pillow image to a sparse matrix mapping.

    Args:
        im: PIL.Image.Image, grayscale image.

    Returns:
        dict: Mapping from (x, y) -> 1/0 with key 'size' storing (w, h).
    """

    w, h = im.size
    data = list(im.getdata())
    mtx = {(i % w, i // w): (1 if data[i] > 200 else 0) for i in range(len(data))}
    mtx['size'] = (w, h)
    return mtx

def mtx2im(mtx,n=255):
    """Convert the sparse matrix mapping back to a Pillow image.

    Args:
        mtx: dict produced by :func:`im2mtx`.
        n: int, grayscale fill value for ink (default 255).

    Returns:
        PIL.Image.Image: L-mode image.
    """

    w, h = mtx['size']
    im = Image.new("L", (w, h))
    dr = ImageDraw.Draw(im)
    for x in range(w):
        for y in range(h):
            dr.point([(x, y)], fill=mtx[x, y] * n)
    return im

def is_chinese_char(uch):
    """Return True if ``uch`` is a CJK or fullwidth character.

    Args:
        uch: str, single Unicode character.

    Returns:
        bool: True when ``uch`` falls into common CJK or fullwidth blocks.
    """

    return u'\u4e00' <= uch <= u'\u9fff' or u'\uff00' <= uch <= u'\uffef'

def rastBox(l, w=100, h=100, f="fonts/Heiti.ttc"):
    """Rasterize a single character into a matrix using a reference scale.

    The function measures a reference glyph to compute a scaling factor so
    that the target character is rendered consistently across sizes.

    Args:
        l: str, single character to rasterize.
        w: int, output image width.
        h: int, output image height.
        f: str, path to TrueType font file.

    Returns:
        dict: Sparse matrix mapping as produced by :func:`im2mtx`.
    """

    im = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(im)

    # 1. Choose a reference character to determine vertical scale.
    if is_chinese_char(l):
        ref_char = "国"
    elif ord(l) < 128 and not l.isalpha():
        # Use brace as a reference for ASCII symbols and punctuation.
        ref_char = "{"
    else:
        ref_char = "Hg"

    # 2. Measure reference ink height using an oversized probe font.
    probe_size = h * 3
    font_probe = ImageFont.truetype(f, probe_size)
    try:
        bbox = font_probe.getbbox(ref_char)
    except AttributeError:
        bbox = dr.textbbox((0, 0), ref_char, font=font_probe)

    ref_ink_height = bbox[3] - bbox[1]

    # 3. Compute scale factor to fit the reference ink height to target height.
    if ref_ink_height == 0:
        scale = 1.0
    else:
        scale = h / float(ref_ink_height)

    # 4. Derive final font size and create font object.
    real_size = int(probe_size * scale)
    font = ImageFont.truetype(f, real_size)

    # 5. Compute vertical offset so the reference glyph's top aligns to y=0.
    try:
        real_bbox = font.getbbox(ref_char)
    except AttributeError:
        real_bbox = dr.textbbox((0, 0), ref_char, font=font)

    y = -real_bbox[1]

    # Horizontal centering: compute content width and center accordingly.
    try:
        wb = font.getbbox(l)
        content_w = wb[2] - wb[0]
        offset_x = wb[0]
    except AttributeError:
        content_w, _ = dr.textsize(l, font=font)
        offset_x = 0

    x = (w - content_w) // 2 - offset_x
    dr.text((x, y), l, font=font, fill=255)

    return im2mtx(im)

