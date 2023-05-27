import math
import sys
import cv2
import pygame
import numpy as np
import re

import standard_library_of_paths


def get_color_values_map(img, cell_size=5):
    height, width = img.shape
    if height != width:
        raise ValueError('height != width')
    elif cell_size >= height:
        raise ValueError('cell_size can not be >= height')
    elif height % cell_size != 0:
        raise ValueError('height must be divided without remainder by cell_size')

    lines_count = height // cell_size
    color_values_map = [[255 for _ in range(lines_count)] for _ in range(lines_count)]

    for i in range(lines_count):
        for j in range(lines_count):
            value = img[i * cell_size:i * cell_size + cell_size, j * cell_size:j * cell_size + cell_size].mean(
                axis=(0, 1))
            color_values_map[i][j] = value

    return color_values_map


def get_bitmap(color_values_map, threshold):
    bitmap_length = len(color_values_map)
    bitmap = [[0 for _ in range(bitmap_length)] for _ in range(bitmap_length)]

    for i in range(bitmap_length):
        for j in range(bitmap_length):
            if color_values_map[i][j] < threshold:
                bitmap[i][j] = 1

    return bitmap


def get_bitmap_with_segments_info(bitmap):
    bitmap_ = [row[:] for row in bitmap]
    bitmap_length = len(bitmap_)

    for i in range(bitmap_length):
        segment_num = 1
        j = 0
        while j < (bitmap_length - 1):
            if bitmap_[i][j] == 1:
                if bitmap_[i][j + 1] == 1:
                    bitmap_[i][j] = segment_num
                    bitmap_[i][j + 1] = segment_num
                    k = j + 2
                    while (k < bitmap_length) and (bitmap_[i][k] == 1):
                        bitmap_[i][k] = segment_num
                        k += 1

                    segment_num += 1
                    j = k + 1
                else:
                    bitmap_[i][j] = segment_num
                    segment_num += 1
                    j += 2
            else:
                j += 1

    return bitmap_


def get_segments_map(bitmap):
    bitmap_ = get_bitmap_with_segments_info(bitmap)
    segments_map = []

    for i in range(len(bitmap_)):
        segments_dict = {}
        segments = np.setdiff1d(np.unique(bitmap_[i]), [0])
        for segment in segments:
            segment_info = {}
            ii = np.where(np.array(bitmap_[i]) == segment)[0]
            segment_info['idx'] = (ii.min(), ii.max())
            segment_info['len'] = ii.max() - ii.min() + 1
            segment_info['mid'] = (ii.max() + ii.min()) // 2
            segments_dict[segment] = segment_info

        segments_map.append(segments_dict)

    return segments_map


def _get_longest_segment(segments_map_row):
    segment_key = None
    if len(segments_map_row.keys()) != 0:
        temp_len = 10e-6
        for key in segments_map_row.keys():
            segment_len = segments_map_row[key]['len']
            if segment_len > temp_len:
                temp_len = segment_len
                segment_key = key

    return segment_key


def _get_suitable_segments(segments_info, segments_map_row):
    suitable_segments = []
    li_c, ri_c = segments_info['idx']

    for key in segments_map_row.keys():
        li_n, ri_n = segments_map_row[key]['idx']
        if not ((li_c < li_n and ri_c < li_n) or li_c > ri_n):
            suitable_segments.append(key)

    if len(suitable_segments) == 0:
        return None
    else:
        return suitable_segments


def _get_next_rows(segments_map, current_row_index):
    next_rows = []

    if current_row_index > (len(segments_map) - 1):
        raise ValueError('current_row_index > (len(segments_map) - 1)')

    for i in range(current_row_index + 1, len(segments_map)):
        next_row = segments_map[i]
        if len(next_row.keys()) != 0:
            next_rows.append((i, next_row))
        else:
            break

    if len(next_rows) != 0:
        return next_rows
    else:
        return None


def _get_segments_to_delete(segments_to_cluster):
    to_delete = {}
    rows_idx = []
    for seg_item in segments_to_cluster:
        rows_idx.append(seg_item['row_idx'])

    rows_idx = np.unique(rows_idx)
    for i in rows_idx:
        delete_list = []
        for seg_item in segments_to_cluster:
            if i == seg_item['row_idx']:
                delete_list.append(seg_item['segment'])

        to_delete[i] = delete_list

    return to_delete


def get_path_clusters(bitmap):
    path_clusters = []
    segments_map = get_segments_map(bitmap)

    for idx, current_row in enumerate(segments_map):
        current_row = (idx, current_row)
        segment = _get_longest_segment(current_row[1])

        if segment is not None:
            segments_to_cluster = [
                {
                    'row_idx': idx,
                    'segment_idx': current_row[1][segment]['idx'],
                    'segment': segment
                }
            ]
            next_rows = _get_next_rows(segments_map, idx)

            if next_rows is not None:
                for next_row in next_rows:
                    idx_n = next_row[0]

                    suitable_segments = _get_suitable_segments(current_row[1][segment], next_row[1])

                    if suitable_segments is not None:
                        suitable_segment = suitable_segments[0]
                        # Find the nearest segment
                        prev_dist = 10e6
                        for suit_seg in suitable_segments:
                            dist = abs(current_row[1][segment]['mid'] - next_row[1][suit_seg]['mid'])
                            if dist < prev_dist:
                                prev_dist = dist
                                suitable_segment = suit_seg

                        segments_to_cluster.append({
                            'row_idx': idx_n,
                            'segment_idx': next_row[1][suitable_segment]['idx'],
                            'segment': suitable_segment
                        })

                        current_row = next_row
                        segment = suitable_segment

                    else:
                        break

            # Delete processed segments from segments map
            to_delete = _get_segments_to_delete(segments_to_cluster)
            for i in range(len(segments_map)):
                delete_list = to_delete.get(i, [])
                for x in delete_list:
                    if x in segments_map[i]:
                        del segments_map[i][x]

            # Add segments cluster to the path clusters
            path_clusters.append(segments_to_cluster)

    return path_clusters, segments_map


def get_motion_paths(path_clusters, color_values_map=None, cell_size=5, is_border_cell=False,
                     paths_for_border_cell=None):
    motion_paths = []
    raw_motion_paths = []
    raw_motion_paths_2 = []

    std_paths = standard_library_of_paths.paths

    for _, path_cluster in enumerate(path_clusters):
        if is_border_cell:
            raw_motion_paths.append('move')
        for segment in path_cluster:
            if not is_border_cell:
                raw_motion_paths.append('move')
            x_left = segment['segment_idx'][0]
            x_right = segment['segment_idx'][1]
            y = segment['row_idx']
            x_current = x_left

            while x_current <= x_right:
                if not is_border_cell:
                    mean_color = color_values_map[y][x_current]
                    density = (255 - mean_color) / 255
                    std_path_key = tuple(std_paths.keys())[0]
                    for key in std_paths.keys():
                        if abs((key / 25) - density) < abs((std_path_key / 25) - density):
                            std_path_key = key

                    std_path = std_paths[std_path_key]
                else:
                    # std_path = std_paths[25]
                    std_path = paths_for_border_cell[y][x_current]

                small_cell_size = cell_size // 5
                coefficient = 0
                x_shift = x_current * cell_size
                y_shift = y * cell_size

                if std_path is not None:
                    for i in range(len(std_path)):
                        point_to_move = (
                            std_path[i][1] * small_cell_size - coefficient + x_shift,
                            std_path[i][0] * small_cell_size - coefficient + y_shift
                        )

                        raw_motion_paths.append(point_to_move)

                x_current += 1

    i = 0
    while i < len(raw_motion_paths):
        if raw_motion_paths[i] == 'move' and (i < (len(raw_motion_paths) - 1)):
            raw_motion_paths_2.append((raw_motion_paths[i + 1], 'move'))
            i += 2
        else:
            raw_motion_paths_2.append((raw_motion_paths[i], 'extrude'))
            i += 1

    i = 0
    while i < len(raw_motion_paths_2) - 1:
        current_point = raw_motion_paths_2[i]
        motion_paths.append(current_point)
        j = i + 1
        next_point = raw_motion_paths_2[j]
        match_type = None
        if current_point[0][0] == next_point[0][0]:
            match_type = 'horizontal'
        elif current_point[0][1] == next_point[0][1]:
            match_type = 'vertical'

        if match_type is not None:
            index_to_check = 0 if match_type == 'horizontal' else 1

            k = j
            while j < len(raw_motion_paths_2):
                if current_point[0][index_to_check] == raw_motion_paths_2[j][0][index_to_check]:
                    k = j
                    j += 1
                else:
                    k = j - 1
                    break

            i = j
            motion_paths.append(raw_motion_paths_2[k])

        else:
            i += 1

    return motion_paths


def x_convert_to_cartesian(x, x_min, x_max, width):
    return x_min + ((x_max - x_min) * x / width)


def y_convert_to_cartesian(y, y_min, y_max, height):
    return y_max - ((y_max - y_min) * y / height)


def get_gcode_file(infill_motion_paths, border_motion_paths, print_options):
    layer_height = print_options['layer_height']
    flow_modifier = print_options['flow_modifier']
    nozzle_diameter = print_options['nozzle_diameter']
    filament_diameter = print_options['filament_diameter']
    offset = print_options['offset']
    layers_count = print_options['layers_count']
    width = print_options['width']
    height = print_options['height']
    file_name = print_options['file_name']

    with open(f'{file_name}_LC_{layers_count}_FW_{flow_modifier}_.gcode', 'w', encoding='utf-8') as f:
        f.write(f'G1 F1200\n')
        for j in range(1, (layers_count + 1)):
            z = layer_height * j
            if j > 1:
                f.write(f'G0 Z{z}\n')

            if len(infill_motion_paths) > 0:
                f.write(';TYPE:Solid infill\n')
                for i in range(len(infill_motion_paths)):
                    path = infill_motion_paths[i]
                    x, y = path[0]
                    x = x_convert_to_cartesian(x, 0, width, width) / 2 + offset
                    y = y_convert_to_cartesian(y, 0, height, height) / 2 + offset
                    action_type = path[1]
                    if action_type == 'move':
                        f.write(f'G1 X{x} Y{y} F9000\n')
                        f.write(f'G1 F1200\n')
                    else:
                        prev_path = infill_motion_paths[i - 1]
                        x_prev, y_prev = prev_path[0]
                        x_prev = x_convert_to_cartesian(x_prev, 0, width, width) / 2 + offset
                        y_prev = y_convert_to_cartesian(y_prev, 0, height, height) / 2 + offset
                        dist = math.dist((x, y), (x_prev, y_prev))
                        E = (4 * layer_height * flow_modifier * nozzle_diameter * dist) / (
                                math.pi * filament_diameter * filament_diameter)
                        f.write(f'G1 X{x} Y{y} E{E}\n')

            if len(border_motion_paths) > 0:
                f.write(';TYPE:Perimeter\n')
                for i in range(len(border_motion_paths)):
                    path = border_motion_paths[i]
                    x, y = path[0]
                    x = x_convert_to_cartesian(x, 0, width, width) / 2 + offset
                    y = y_convert_to_cartesian(y, 0, height, height) / 2 + offset
                    action_type = path[1]
                    if action_type == 'move':
                        f.write(f'G1 X{x} Y{y} F9000\n')
                        f.write(f'G1 F1200\n')
                    else:
                        prev_path = border_motion_paths[i - 1]
                        x_prev, y_prev = prev_path[0]
                        x_prev = x_convert_to_cartesian(x_prev, 0, width, width) / 2 + offset
                        y_prev = y_convert_to_cartesian(y_prev, 0, height, height) / 2 + offset
                        dist = math.dist((x, y), (x_prev, y_prev))
                        E = (4 * layer_height * flow_modifier * nozzle_diameter * dist) / (
                                math.pi * filament_diameter * filament_diameter)
                        f.write(f'G1 X{x} Y{y} E{E}\n')


def draw_border(ctx, img, cell_size=5, white_pixel_threshold=254):
    height, width = img.shape
    lines_count = height // cell_size

    for i in range(lines_count):
        for j in range(lines_count):
            value = img[i * cell_size:i * cell_size + cell_size, j * cell_size:j * cell_size + cell_size].mean(
                axis=(0, 1))

            left = j * cell_size
            top = i * cell_size
            _width = cell_size
            _height = cell_size

            if value >= white_pixel_threshold:
                color = (255, 0, 0)
                pygame.draw.rect(ctx, color, (left, top, _width, _height))
            else:
                for k in range(0, cell_size + 1):
                    for m in range(0, cell_size + 1):
                        value = img[i * cell_size + k, j * cell_size + m]

                        if value >= white_pixel_threshold:
                            color = (0, 255, 0)
                            pygame.draw.rect(ctx, color, (left, top, _width, _height))
                            break


def get_bitmap_for_border(img, cell_size=5, white_pixel_threshold=254):
    height, width = img.shape
    if height != width:
        raise ValueError('height != width')
    elif cell_size >= height:
        raise ValueError('cell_size can not be >= height')
    elif height % cell_size != 0:
        raise ValueError('height must be divided without remainder by cell_size')

    lines_count = height // cell_size
    bitmap = [[0 for _ in range(lines_count)] for _ in range(lines_count)]

    for i in range(lines_count):
        for j in range(lines_count):
            value = img[i * cell_size:i * cell_size + cell_size, j * cell_size:j * cell_size + cell_size].mean(
                axis=(0, 1))

            if value < white_pixel_threshold:
                is_border_cell = False
                k = 0
                while (not is_border_cell) and ((i * cell_size + k) < width) and k <= cell_size:
                    m = 0
                    while ((j * cell_size + m) < width) and (m <= cell_size):
                        _value = img[i * cell_size + k, j * cell_size + m]
                        m += 1
                        if _value >= white_pixel_threshold:
                            is_border_cell = True
                            break
                    k += 1

                if is_border_cell:
                    bitmap[i][j] = 1
                else:
                    bitmap[i][j] = 0

    return bitmap


def get_extended_bitmap(img, bitmap, cell_size=5, black_pixel_threshold=30):
    lines_count = len(bitmap)
    extended_bitmap = [[None for _ in range(lines_count)] for _ in range(lines_count)]

    for i in range(lines_count):
        for j in range(lines_count):
            if bitmap[i][j] == 1:
                cell_matrix = [[0 for _ in range(cell_size)] for _ in range(cell_size)]
                for k in range(cell_size):
                    for m in range(cell_size):
                        value = img[i * cell_size + k, j * cell_size + m]
                        if value <= black_pixel_threshold:
                            cell_matrix[k][m] = 1
                        else:
                            cell_matrix[k][m] = 0

                cell_matrix_sum = 0
                for k in range(cell_size):
                    for m in range(cell_size):
                        cell_matrix_sum += cell_matrix[k][m]

                extended_bitmap[i][j] = cell_matrix if cell_matrix_sum >= 5 else None

    return extended_bitmap


def get_paths_for_border_cells(extended_bitmap):
    lines_count = len(extended_bitmap)
    paths = [[None for _ in range(lines_count)] for _ in range(lines_count)]

    for i in range(lines_count):
        for j in range(lines_count):
            if extended_bitmap[i][j] is not None:
                cell_size = len(extended_bitmap[i][j])
                temp_result = []
                for k in range(cell_size):
                    match_indexes = [index for (index, item) in enumerate(extended_bitmap[i][j][k]) if item == 1]
                    start_stop_indexes = None
                    if len(match_indexes) > 1:
                        start_stop_indexes = [match_indexes[0], match_indexes[-1]]

                    if start_stop_indexes is not None:
                        coefficient = 1
                        temp_result.append((k + coefficient, start_stop_indexes[0] + coefficient))
                        temp_result.append((k + coefficient, start_stop_indexes[1] + coefficient))

                paths[i][j] = tuple(temp_result)

    return paths


# Test functions
def test_for_get_color_values_map(ctx, color_values_map, cell_size=5):
    for i in range(len(color_values_map)):
        for j in range(len(color_values_map)):
            color = (color_values_map[i][j], color_values_map[i][j], color_values_map[i][j])
            pygame.draw.rect(ctx, color, (j * cell_size, i * cell_size, cell_size, cell_size))


def test_for_get_bitmap(ctx, bitmap, cell_size=5):
    for i in range(len(bitmap)):
        for j in range(len(bitmap)):
            color = (255, 255, 255) if bitmap[i][j] == 0 else (0, 0, 0)
            pygame.draw.rect(ctx, color, (j * cell_size, i * cell_size, cell_size, cell_size))


def test_for_get_bitmap_with_segments_info(ctx, bitmap, cell_size=5):
    colors_count = 0
    for i in range(len(bitmap)):
        cc = len(np.unique(bitmap[i]))
        if cc > colors_count:
            colors_count = cc

    colors = [(255, 255, 255)]
    for _ in range(colors_count):
        colors.append(np.random.choice(range(255), size=3))

    for i in range(len(bitmap)):
        for j in range(len(bitmap)):
            color_index = bitmap[i][j]
            pygame.draw.rect(ctx, colors[color_index], (j * cell_size, i * cell_size, cell_size, cell_size))


def test_for_get_path_clusters(ctx, path_clusters, cell_size=5):
    for idx, path_cluster in enumerate(path_clusters):
        color = np.random.choice(range(255), size=3)

        for segment in path_cluster:
            left = segment['segment_idx'][0] * cell_size
            top = segment['row_idx'] * cell_size
            width = (segment['segment_idx'][1] - segment['segment_idx'][0] + 1) * cell_size
            height = cell_size
            pygame.draw.rect(ctx, color, (left, top, width, height))


def test_for_get_motion_paths(ctx, motion_paths, color=(101, 142, 196)):
    for i in range(len(motion_paths) - 1):
        if motion_paths[i + 1][1] != 'move':
            from_ = motion_paths[i][0]
            to = motion_paths[i + 1][0]
            pygame.draw.line(ctx, color, from_, to, 1)


def test_for_get_extended_bitmap(ctx, extended_bitmap, cell_size=5):
    for i in range(len(extended_bitmap)):
        for j in range(len(extended_bitmap)):
            if extended_bitmap[i][j] is not None:
                for k in range(cell_size):
                    for m in range(cell_size):
                        if extended_bitmap[i][j][k][m] == 1:
                            top = i * cell_size + k
                            left = j * cell_size + m
                            pygame.draw.rect(ctx, (0, 0, 0), (left, top, 1, 1))


def test_for_get_paths_for_border_cells(ctx, paths_for_border, cell_size=5):
    for i in range(len(paths_for_border)):
        for j in range(len(paths_for_border)):
            if paths_for_border[i][j] is not None:
                for k in range(len(paths_for_border[i][j]) - 1):
                    x = j * cell_size
                    y = i * cell_size
                    start = paths_for_border[i][j][k]
                    start = (x + start[1], y + start[0])
                    stop = paths_for_border[i][j][k + 1]
                    stop = (x + stop[1], y + stop[0])
                    pygame.draw.line(ctx, (0, 0, 0), start, stop)


def get_trajectories_for_cells(ctx, color_values_map, bitmap, cell_size=5):
    std_paths = standard_library_of_paths.paths
    for i in range(len(color_values_map)):
        for j in range(len(color_values_map)):
            if bitmap[i][j] == 1:
                mean_color = color_values_map[i][j]
                density = 1 - (mean_color / 255)
                std_path_key = tuple(std_paths.keys())[0]
                for key in std_paths.keys():
                    if abs((key / 25) - density) < abs((std_path_key / 25) - density):
                        std_path_key = key

                std_path = std_paths[std_path_key]

                small_cell_size = cell_size // 5
                coefficient = 0
                x_shift = j * cell_size
                y_shift = i * cell_size

                for k in range(len(std_path) - 1):
                    from_ = (
                        std_path[k][1] * small_cell_size - coefficient + x_shift,
                        std_path[k][0] * small_cell_size - coefficient + y_shift
                    )
                    to = (
                        std_path[k + 1][1] * small_cell_size - coefficient + x_shift,
                        std_path[k + 1][0] * small_cell_size - coefficient + y_shift
                    )
                    color = (0, 0, 0)
                    pygame.draw.line(ctx, color, from_, to, 2)


def tests():
    # Pygame Settings
    pygame.init()
    canvas = pygame.display.set_mode((300, 300))
    pygame.display.set_caption('Canvas. Tests')
    ctx = canvas
    ctx.set_alpha(None)
    ctx.set_colorkey(None)
    ctx.fill((255, 255, 255))

    # Common part
    infill_img_path = 'images/5_infill.jpg'
    infill_img = cv2.imread(infill_img_path, cv2.IMREAD_GRAYSCALE)
    border_img_path = 'images/5_border.jpg'
    border_img = cv2.imread(border_img_path, cv2.IMREAD_GRAYSCALE)
    height, width = infill_img.shape
    cell_size = 5

    # Infill
    threshold = 254
    color_values_map = get_color_values_map(infill_img, cell_size)
    bitmap = get_bitmap(color_values_map, threshold)
    bitmap_with_segments_info = get_bitmap_with_segments_info(bitmap)
    segments_map = get_segments_map(bitmap)
    path_clusters, _ = get_path_clusters(bitmap)
    motion_paths = get_motion_paths(path_clusters, color_values_map, cell_size)
    # Infill Tests
    # get_trajectories_for_cells(ctx, color_values_map, bitmap, cell_size)
    # test_for_get_color_values_map(ctx, color_values_map, cell_size)
    # test_for_get_bitmap(ctx, bitmap, cell_size)
    # test_for_get_bitmap_with_segments_info(ctx, bitmap_with_segments_info, cell_size)
    # test_for_get_path_clusters(ctx, path_clusters, cell_size)
    # test_for_get_motion_paths(ctx, motion_paths, (0, 0, 0))

    # Border
    # draw_border(ctx, border_img, cell_size)
    border_bitmap = get_bitmap_for_border(border_img)
    border_bitmap_with_segments_info = get_bitmap_with_segments_info(border_bitmap)
    extended_bitmap = get_extended_bitmap(border_img, border_bitmap)
    paths_for_border_cells = get_paths_for_border_cells(extended_bitmap)
    border_path_clusters, _ = get_path_clusters(border_bitmap)
    border_motion_paths = get_motion_paths(border_path_clusters, None, cell_size, True,
                                           paths_for_border_cells)
    # Border Tests
    # test_for_get_bitmap(ctx, border_bitmap)
    # test_for_get_bitmap_with_segments_info(ctx, border_bitmap_with_segments_info, cell_size)
    # test_for_get_path_clusters(ctx, border_path_clusters)
    # test_for_get_extended_bitmap(ctx, extended_bitmap, cell_size)
    # test_for_get_paths_for_border_cells(ctx, paths_for_border_cells)
    # test_for_get_motion_paths(ctx, border_motion_paths, (0, 0, 0))

    # GCODE
    file_name = ''
    # file_name = re.search(r'\d+_small', infill_img_path).group(0)

    print_options = {
        'layer_height': 0.2,
        'flow_modifier': 1,
        'nozzle_diameter': 0.4,
        'filament_diameter': 1.75,
        'offset': 20,
        'layers_count': 50,
        'width': width,
        'height': height,
        'file_name': f'img_{file_name}'
    }
    # get_gcode_file(motion_paths, border_motion_paths, print_options)

    print('debug')

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                print(x, y)

        pygame.display.flip()


def main():
    tests()


main()
