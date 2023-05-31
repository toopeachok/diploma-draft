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


def get_bitmap(color_values_map, threshold=255):
    bitmap_length = len(color_values_map)
    bitmap = [[0 for _ in range(bitmap_length)] for _ in range(bitmap_length)]

    for i in range(bitmap_length):
        for j in range(bitmap_length):
            if color_values_map[i][j] < threshold:
                bitmap[i][j] = 1

    return bitmap


def get_segments_list(bitmap):
    segments_list = []

    i = 0
    while i < len(bitmap):
        j = 0
        while j < len(bitmap):
            if bitmap[i][j] == 1:
                start = j

                k = j + 1
                while (k < len(bitmap)) and (bitmap[i][k] == 1):
                    k += 1

                stop = k - 1
                segments_list.append((i, start, stop))

                j = k + 1
            else:
                j += 1

        i += 1

    return segments_list


def next_segment_with_distance_between_segments(first_segment, second_segment):
    if (first_segment[0] != second_segment[0]) and (first_segment[1] < first_segment[2]):
        second_segment = (second_segment[0], second_segment[2], second_segment[1])

    return second_segment, ((first_segment[0] - second_segment[0]) ** 2 + (
            first_segment[2] - second_segment[1]) ** 2)


def get_closest_segment(current_segment, segments_list):
    if len(segments_list) > 0:
        closest_segment = segments_list[0]
        closest_segment, closest_dist = next_segment_with_distance_between_segments(current_segment, closest_segment)

        for idx, segment in enumerate(segments_list):
            if idx > 0:
                segment, dist = next_segment_with_distance_between_segments(current_segment, segment)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_segment = segment

        return closest_segment

    else:
        return None


def get_path_for_traversing_grid(bitmap):
    path = []
    segments_list = get_segments_list(bitmap)
    segments_list_copy = segments_list[:]
    indexes_to_delete = []
    segment = segments_list[0]
    idx = 0

    while len(segments_list_copy) > 0:
        path.append(segment)
        indexes_to_delete.append(idx)
        segments_list_copy = [elem for i, elem in enumerate(segments_list) if i not in indexes_to_delete]
        if len(segments_list_copy) == 0:
            break

        new_segment_to_add = get_closest_segment(segment, segments_list_copy)

        if new_segment_to_add is not None:
            segment = new_segment_to_add

            segment_to_delete = new_segment_to_add
            if new_segment_to_add[1] > new_segment_to_add[2]:
                segment_to_delete = (new_segment_to_add[0], new_segment_to_add[2], new_segment_to_add[1])
            idx = segments_list.index(segment_to_delete)
        else:
            break

    return path


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


# Test functions
def test_for_get_color_values_map(screen, color_values_map, cell_size=5):
    for i in range(len(color_values_map)):
        for j in range(len(color_values_map)):
            color = (color_values_map[i][j], color_values_map[i][j], color_values_map[i][j])
            pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size))


def test_for_get_bitmap(screen, bitmap, cell_size=5):
    for i in range(len(bitmap)):
        for j in range(len(bitmap)):
            color = (255, 255, 255) if bitmap[i][j] == 0 else (0, 0, 0)
            pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size))


def get_trajectories_for_cells(screen, color_values_map, bitmap, cell_size=5):
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
                    pygame.draw.line(screen, color, from_, to, 2)


def tests():
    # Pygame Settings
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('Tests')
    screen.fill((255, 255, 255))

    # Common part
    infill_img_path = 'images/9.jpg'
    infill_img = cv2.imread(infill_img_path, cv2.IMREAD_GRAYSCALE)
    border_img_path = 'images/5_border.jpg'
    border_img = cv2.imread(border_img_path, cv2.IMREAD_GRAYSCALE)
    height, width = infill_img.shape
    cell_size = 5

    img = pygame.image.load(infill_img_path)
    screen.blit(img, (0, 0))

    # Infill
    threshold = 254
    color_values_map = get_color_values_map(infill_img, cell_size)
    bitmap = get_bitmap(color_values_map, threshold)
    segments_list = get_segments_list(bitmap)
    path_for_traversing_grid = get_path_for_traversing_grid(bitmap)

    # Infill Tests
    # get_trajectories_for_cells(screen, color_values_map, bitmap, cell_size)
    test_for_get_color_values_map(screen, color_values_map, cell_size)
    # test_for_get_bitmap(screen, bitmap, cell_size)

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
    # get_gcode_file([], [], print_options)

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
