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


def get_segments_map(bitmap):
    bitmap_ = _get_bitmap_with_segments_info(bitmap)
    segments_map = []

    bitmap_length = len(bitmap_)
    for i in range(bitmap_length):
        segments_dict = {}
        segments = np.delete(np.unique(bitmap_[i]), [0])
        for segment in segments:
            segment_info = {}
            ii = np.where(np.array(bitmap_[i]) == segment)[0]
            segment_info['indexes'] = (ii.min(), ii.max())
            segment_info['length'] = ii.max() - ii.min() + 1
            segment_info['mid'] = (ii.max() + ii.min()) // 2
            segments_dict[segment] = segment_info

        segments_map.append(segments_dict)

    return segments_map


def tests():
    pygame.init()
    canvas = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption('Canvas. Tests')
    ctx = canvas
    ctx.set_alpha(None)
    ctx.set_colorkey(None)
    ctx.fill((255, 255, 255))

    img_path = 'images/9.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cell_size = 10

    thresholds_map = get_thresholds_map(img, cell_size)

    threshold_coefficient = 1
    threshold = img.mean(axis=(0, 1)) // threshold_coefficient
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
        for i in range(len(bitmap_with_segments_info_)):
            for j in range(len(bitmap_with_segments_info_)):
                colors = (
                    (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255))
                color_index = bitmap_with_segments_info_[i][j]
                pygame.draw.rect(ctx, colors[color_index], (j * cell_size, i * cell_size, cell_size, cell_size))

    # test_for_get_thresholds_map(thresholds_map)
    # test_for_get_bitmap(bitmap)
    # test_for_get_bitmap_with_segments_info(bitmap_with_segments_info)

    segments_map = get_segments_map(bitmap)
    print(segments_map)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.display.flip()


def main():
    tests()


main()
