import sys
import cv2
import pygame
import numpy as np


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


def _get_bitmap_with_row_connectivity_segments(bitmap):
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


def get_connectivity_segments_list(bitmap):
    bitmap_ = _get_bitmap_with_row_connectivity_segments(bitmap)
    bitmap_length = len(bitmap_)
    connectivity_segments = [None for _ in range(bitmap_length)]
    connectivity_segments_list = []

    for i in range(bitmap_length):
        values = np.unique(bitmap_[i])
        values = np.delete(values, [0])
        segments_count = len(values)

        if segments_count == 0:
            continue
        else:
            segments = [None for _ in range(segments_count)]
            # noinspection PyTypeChecker
            connectivity_segments[i] = segments

    for i in range(bitmap_length):
        if connectivity_segments[i] is None:
            continue
        else:
            # noinspection PyTypeChecker
            for j in range(len(connectivity_segments[i])):
                segment_num = j + 1
                # print(f'segment_num: {segment_num}')
                bitmap_row = np.array(bitmap_[i])
                # print(f'bitmap_row: {bitmap_row}')
                ii = np.where(bitmap_row == segment_num)[0]
                # print(f'ii: {ii}')
                # noinspection PyUnresolvedReferences
                connectivity_segments[i][j] = (ii.min(), ii.max())

    i = 0
    while i < (bitmap_length - 1):
        if connectivity_segments[i] is None:
            i += 1
            continue
        else:
            current_row_segments_count = len(connectivity_segments[i])
            j = i
            while (j < bitmap_length - 1) and (connectivity_segments[j + 1] is not None) and (
                    len(connectivity_segments[j + 1]) == current_row_segments_count):
                j += 1

            if j > i:
                temp = []
                for k in range(i, j + 1):
                    temp.append((k, connectivity_segments[k]))
                    # print(k, connectivity_segments[k])

                connectivity_segments_list.append(temp)

                i = j
            else:
                i += 1

            # print(f'i: {i}, j: {j}')

    # print(connectivity_segments_list)
    return connectivity_segments_list


def get_clusters_from_connectivity_segments_list(connectivity_segments_list):
    connectivity_segments_list_ = [row[:] for row in connectivity_segments_list]
    # print(connectivity_segments_list_)
    clusters = []
    for i in range(len(connectivity_segments_list_)):
        # [(row_index, [(col_index_min, col_index_max)]), (row_index, [(col_index_min, col_index_max)])]
        # connectivity_segments_list_[i]
        # [(11, [(39, 40)]), (12, [(39, 41)])]
        # connectivity_segments_list_[i][j]
        # (11, [(39, 40)])
        current_segments_sublist = connectivity_segments_list_[i][0][1]
        segments_count_in_sublist = len(current_segments_sublist)
        # print(f'i: {i}, segments_count_in_sublist: {segments_count_in_sublist}')
        # print(f'i: {i}, current_segments_sublist: {current_segments_sublist}')
        # print(f'i: {i}, len connectivity_segments_list_[i]: {len(connectivity_segments_list_[i])}')
        # print(f'i: {i}, connectivity_segments_list_[i]: {connectivity_segments_list_[i]}')

        if segments_count_in_sublist == 1:
            result = []
            for j in range(len(connectivity_segments_list_[i])):
                segment_info = connectivity_segments_list_[i][j]
                row_index = segment_info[0]
                segment_indexes = segment_info[1][0]
                result.append((row_index, segment_indexes[0], segment_indexes[1]))

            # print(f'result : {result}')
            clusters.append(result)

    return clusters


def tests():
    pygame.init()
    canvas = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption('Canvas. Tests')
    ctx = canvas
    ctx.set_alpha(None)
    ctx.set_colorkey(None)
    ctx.fill((255, 255, 255))

    img_path = 'images/3.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cell_size = 10

    thresholds_map = get_thresholds_map(img, cell_size)

    threshold_coefficient = 1
    threshold = img.mean(axis=(0, 1)) // threshold_coefficient
    bitmap = get_bitmap(thresholds_map, threshold)
    _bitmap_with_connectivity_segments = _get_bitmap_with_row_connectivity_segments(bitmap)

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

    def test_for_get_bitmap_with_row_connectivity_segments(bitmap_with_connectivity_segments_):
        for i in range(len(bitmap_with_connectivity_segments_)):
            for j in range(len(bitmap_with_connectivity_segments_)):
                colors = (
                    (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255))
                color_index = bitmap_with_connectivity_segments_[i][j]
                pygame.draw.rect(ctx, colors[color_index], (j * cell_size, i * cell_size, cell_size, cell_size))

    def test_for_get_clusters_from_connectivity_segments_list(clusters_):
        for i in range(len(clusters_)):
            for segment_info in clusters[i]:
                color = (25 * i, 25 * i, 25 * i)
                left = segment_info[1] * cell_size
                top = segment_info[0] * cell_size
                width = (segment_info[2] - segment_info[1]) * cell_size + cell_size
                height = cell_size
                pygame.draw.rect(ctx, color, (left, top, width, height))

    # test_for_get_thresholds_map(thresholds_map)
    # test_for_get_bitmap(bitmap)
    test_for_get_bitmap_with_row_connectivity_segments(_bitmap_with_connectivity_segments)
    connectivity_segments_list = get_connectivity_segments_list(bitmap)
    # print(connectivity_segments_list)
    clusters = get_clusters_from_connectivity_segments_list(connectivity_segments_list)
    test_for_get_clusters_from_connectivity_segments_list(clusters)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()


def main():
    tests()


main()
