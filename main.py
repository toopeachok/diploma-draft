import sys
import cv2
import pygame
import numpy as np

import standard_library_of_paths


def get_thresholds_map(img, cell_size=10):
    height, width = img.shape
    if height != width:
        raise ValueError('height != width')
    elif cell_size >= height:
        raise ValueError('cell_size can not be >= height')
    elif height % cell_size != 0:
        raise ValueError('height must be divided without remainder by cell_size')

    lines_count = height // cell_size
    thresholds_map = [[0 for _ in range(lines_count)] for _ in range(lines_count)]

    for i in range(lines_count):
        for j in range(lines_count):
            value = img[i * cell_size:i * cell_size + cell_size, j * cell_size:j * cell_size + cell_size].mean(
                axis=(0, 1))
            thresholds_map[i][j] = value

    return thresholds_map


def get_bitmap(thresholds_map, threshold):
    bitmap_length = len(thresholds_map)
    bitmap = [[0 for _ in range(bitmap_length)] for _ in range(bitmap_length)]

    for i in range(bitmap_length):
        for j in range(bitmap_length):
            if thresholds_map[i][j] <= threshold:
                bitmap[i][j] = 1

    return bitmap


def _get_bitmap_with_segments_info(bitmap):
    bitmap_result = [row[:] for row in bitmap]
    bitmap_length = len(bitmap_result)

    for i in range(bitmap_length):
        segment_num = 1
        j = 0
        while j < (bitmap_length - 1):
            if bitmap_result[i][j] == 1:
                if bitmap_result[i][j + 1] == 1:
                    bitmap_result[i][j] = segment_num
                    bitmap_result[i][j + 1] = segment_num
                    k = j + 2
                    while (k < bitmap_length) and (bitmap_result[i][k] == 1):
                        bitmap_result[i][k] = segment_num
                        k += 1

                    segment_num += 1
                    j = k + 1
                else:
                    bitmap_result[i][j] = segment_num
                    segment_num += 1
                    j += 2
            else:
                j += 1

    return bitmap_result


def _get_segments_map(bitmap):
    bitmap_ = _get_bitmap_with_segments_info(bitmap)
    segments_map = []

    bitmap_length = len(bitmap_)
    for i in range(bitmap_length):
        segments_dict = {}
        segments = np.delete(np.unique(bitmap_[i]), [0])
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
    _segment = None
    if len(segments_map_row.keys()) != 0:
        _temp = 0
        for _key in segments_map_row.keys():
            _segment_len = segments_map_row[_key]['len']
            if _segment_len > _temp:
                _temp = _segment_len
                _segment = _key

    return _segment


def _get_suitable_segments(segments_info, segments_map_row):
    _suitable_segments = []
    li_c, ri_c = segments_info['idx']

    for key in segments_map_row.keys():
        li_n, ri_n = segments_map_row[key]['idx']
        if ~((li_c < li_n and ri_c < li_n) or li_c > ri_n):
            _suitable_segments.append(key)

    if len(_suitable_segments) == 0:
        return None
    else:
        return _suitable_segments


def _get_next_rows(segments_map, current_row_index):
    next_rows = []

    if current_row_index > (len(segments_map) - 1):
        raise ValueError('current_row_index > (len(segments_map) - 1)')

    for i in range(current_row_index + 1, len(segments_map)):
        _next_row = segments_map[i]
        if len(_next_row.keys()) != 0:
            next_rows.append((i, _next_row))
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
    segments_map = _get_segments_map(bitmap)

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


def get_moving_paths(path_clusters, thresholds_map, cell_size):
    moving_paths = []
    raw_moving_paths = []
    raw_moving_paths_2 = []

    std_paths = standard_library_of_paths.paths

    for idx, path_cluster in enumerate(path_clusters):
        for segment in path_cluster:
            raw_moving_paths.append('move')
            x_left = segment['segment_idx'][0]
            x_right = segment['segment_idx'][1]
            y = segment['row_idx']
            x_current = x_left

            while x_current <= x_right:
                mean_color = thresholds_map[y][x_current]
                density = (255 - mean_color) / 255
                std_path_key = tuple(std_paths.keys())[0]
                for key in std_paths.keys():
                    if abs((key / 25) - density) < abs((std_path_key / 25) - density):
                        std_path_key = key

                std_path = std_paths[std_path_key]

                small_cell_size = cell_size // 5
                coefficient = 1
                x_shift = x_current * cell_size
                y_shift = y * cell_size

                for i in range(len(std_path)):
                    point_to_move = (
                        std_path[i][1] * small_cell_size - coefficient + x_shift,
                        std_path[i][0] * small_cell_size - coefficient + y_shift
                    )

                    raw_moving_paths.append(point_to_move)

                x_current += 1

    i = 0
    while i < len(raw_moving_paths):
        if raw_moving_paths[i] == 'move' and (i < (len(raw_moving_paths) - 1)):
            raw_moving_paths_2.append((raw_moving_paths[i + 1], 'move'))
            i += 2
        else:
            raw_moving_paths_2.append((raw_moving_paths[i], 'extrude'))
            i += 1

    i = 0
    while i < len(raw_moving_paths_2) - 1:
        current_point = raw_moving_paths_2[i]
        moving_paths.append(current_point)
        j = i + 1
        next_point = raw_moving_paths_2[j]
        match_type = None
        if current_point[0][0] == next_point[0][0]:
            match_type = 'horizontal'
        elif current_point[0][1] == next_point[0][1]:
            match_type = 'vertical'

        if match_type is not None:
            index_to_check = 0 if match_type == 'horizontal' else 1

            k = j
            while j < len(raw_moving_paths_2):
                if current_point[0][index_to_check] == raw_moving_paths_2[j][0][index_to_check]:
                    k = j
                    j += 1
                else:
                    k = j - 1
                    break

            i = j
            moving_paths.append(raw_moving_paths_2[k])

        else:
            i += 1

    return moving_paths


def x_convert_to_cartesian(x, x_min, x_max, width):
    return x_min + ((x_max - x_min) * x / width)


def y_convert_to_cartesian(y, y_min, y_max, height):
    return y_max - ((y_max - y_min) * y / height)


def tests():
    pygame.init()
    canvas = pygame.display.set_mode((400, 400))
    pygame.display.set_caption('Canvas. Tests')
    ctx = canvas
    ctx.set_alpha(None)
    ctx.set_colorkey(None)
    ctx.fill((255, 255, 255))

    img_path = 'images/6.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cell_size = 5

    thresholds_map = get_thresholds_map(img, cell_size)
    threshold = 240
    bitmap = get_bitmap(thresholds_map, threshold)
    bitmap_with_segments_info = _get_bitmap_with_segments_info(bitmap)

    def test_for_get_thresholds_map(thresholds_map_):
        for i in range(len(thresholds_map_)):
            for j in range(len(thresholds_map_)):
                color = (thresholds_map_[i][j], thresholds_map_[i][j], thresholds_map_[i][j])
                pygame.draw.rect(ctx, color, (j * cell_size, i * cell_size, cell_size, cell_size))

    def test_for_get_bitmap(bitmap_):
        for i in range(len(bitmap_)):
            for j in range(len(bitmap_)):
                color = (255, 255, 255) if bitmap_[i][j] == 0 else (0, 0, 0)
                pygame.draw.rect(ctx, color, (j * cell_size, i * cell_size, cell_size, cell_size))

    def test_for_get_bitmap_with_segments_info(bitmap_with_segments_info_):
        colors_count = 0
        for i in range(len(bitmap_with_segments_info_)):
            cc = len(np.unique(bitmap_with_segments_info_[i]))
            if cc > colors_count:
                colors_count = cc

        colors = [(255, 255, 255)]
        for _ in range(colors_count):
            colors.append(np.random.choice(range(255), size=3))

        for i in range(len(bitmap_with_segments_info_)):
            for j in range(len(bitmap_with_segments_info_)):
                color_index = bitmap_with_segments_info_[i][j]
                pygame.draw.rect(ctx, colors[color_index], (j * cell_size, i * cell_size, cell_size, cell_size))

    def test_for_get_path_clusters(path_clusters_):
        for idx, path_cluster in enumerate(path_clusters_):
            color = np.random.choice(range(255), size=3)

            for segment in path_cluster:
                left = segment['segment_idx'][0] * cell_size
                top = segment['row_idx'] * cell_size
                width = (segment['segment_idx'][1] - segment['segment_idx'][0] + 1) * cell_size
                height = cell_size
                pygame.draw.rect(ctx, color, (left, top, width, height))

    def test_for_get_moving_paths(moving_paths_):
        for i in range(len(moving_paths_) - 1):
            if moving_paths_[i + 1][1] != 'move':
                from_ = moving_paths_[i][0]
                to = moving_paths_[i + 1][0]
                color = (101, 142, 196)
                pygame.draw.line(ctx, color, from_, to, 1)
                # pygame.draw.circle(ctx, (225, 18, 0), to, 1, 1)

    # test_for_get_thresholds_map(thresholds_map)
    # test_for_get_bitmap(bitmap)
    # test_for_get_bitmap_with_segments_info(bitmap_with_segments_info)

    segments_map = _get_segments_map(bitmap)
    path_clusters, _segments_map_ = get_path_clusters(bitmap)

    # test_for_get_path_clusters(path_clusters)

    moving_paths = get_moving_paths(path_clusters, thresholds_map, cell_size)

    with open('myGCode.gcode', 'w') as f:
        for path in moving_paths:
            x, y = path[0]
            x = x_convert_to_cartesian(x, -110, 110, 400) / 2
            y = y_convert_to_cartesian(y, -110, 110, 400) / 2
            action_type = path[1]
            if action_type == 'move':
                f.write(f'G0 X{x} Y{y} ;\n')
            else:
                f.write(f'G1 X{x} Y{y} ;\n')

    test_for_get_moving_paths(moving_paths)

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
