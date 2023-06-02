import math
import sys
import cv2
import pygame
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


def get_printing_path(bitmap, color_values_map=None, cell_size=5):
    printing_path = []
    std_paths = standard_library_of_paths.paths

    path_for_traversing_grid = get_path_for_traversing_grid(bitmap)
    for continuous_path_segment in path_for_traversing_grid:
        direction = 1 if continuous_path_segment[1] < continuous_path_segment[2] else -1
        printing_path_for_cps = []

        for i in range(continuous_path_segment[1], continuous_path_segment[2] + direction, direction):
            y = continuous_path_segment[0]
            x = i
            if color_values_map is not None:
                mean_color = color_values_map[y][x]
                density = 1 - mean_color / 255
                std_path_key = tuple(std_paths.keys())[0]
                for key in std_paths.keys():
                    if abs(key / 25 - density) < abs(std_path_key / 25 - density):
                        std_path_key = key

                std_path = std_paths[std_path_key]
            else:
                std_path = std_paths[25]

            small_cell_size = cell_size // 5
            coefficient = 1
            x_shift = x * cell_size
            y_shift = y * cell_size

            cell_points = []
            for j in range(len(std_path)):
                point_to_move = (
                    std_path[j][1] * small_cell_size - coefficient + x_shift,
                    std_path[j][0] * small_cell_size - coefficient + y_shift
                )
                cell_points.append(point_to_move)

            cell_points = cell_points[::direction]
            printing_path_for_cps = printing_path_for_cps + cell_points

        printing_path.append(printing_path_for_cps)

    return printing_path


# Border
def draw_border(screen, img, cell_size=5, white_pixel_threshold=254):
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
                pygame.draw.rect(screen, color, (left, top, _width, _height))
            else:
                for k in range(0, cell_size + 1):
                    for m in range(0, cell_size + 1):
                        value = img[i * cell_size + k, j * cell_size + m]

                        if value >= white_pixel_threshold:
                            color = (0, 255, 0)
                            pygame.draw.rect(screen, color, (left, top, _width, _height))
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

    return bitmap


def get_extended_bitmap(img, bitmap, cell_size=5, black_pixel_threshold=30):
    lines_count = len(bitmap)
    extended_bitmap = [[0 for _ in range(lines_count)] for _ in range(lines_count)]

    for i in range(lines_count):
        for j in range(lines_count):
            if bitmap[i][j] == 1:
                cell_matrix = [[0 for _ in range(cell_size)] for _ in range(cell_size)]
                for k in range(cell_size):
                    for m in range(cell_size):
                        value = img[i * cell_size + k, j * cell_size + m]
                        if value < black_pixel_threshold:
                            cell_matrix[k][m] = 1
                        else:
                            cell_matrix[k][m] = 0

                cell_matrix_sum = 0
                for k in range(cell_size):
                    for m in range(cell_size):
                        cell_matrix_sum += cell_matrix[k][m]

                if cell_matrix_sum >= 5:
                    extended_bitmap[i][j] = cell_matrix

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


# G-code
def x_convert_to_cartesian(x, x_min, x_max, width):
    return x_min + ((x_max - x_min) * x / width)


def y_convert_to_cartesian(y, y_min, y_max, height):
    return y_max - ((y_max - y_min) * y / height)


def get_gcode_file(print_options, infill_printing_path=(), border_printing_path=()):
    layer_height = print_options['layer_height']
    flow_modifier = print_options['flow_modifier']
    nozzle_diameter = print_options['nozzle_diameter']
    filament_diameter = print_options['filament_diameter']
    offset = print_options['offset']
    layers_count = print_options['layers_count']
    width = print_options['width']
    height = print_options['height']
    file_name = print_options['file_name']
    file_helper_name = print_options['file_helper_name']
    file_lines = []

    if len(infill_printing_path) > 0 and len(border_printing_path) > 0:
        file_name = file_name + '_total'
    elif len(infill_printing_path) > 0:
        file_name = file_name + '_infill'
    elif len(border_printing_path) > 0:
        file_name = file_name + '_border'

    with open(file_helper_name) as f:
        file_helper_lines = f.readlines()

    my_code_start_line_index = file_helper_lines.index('; my code start\n')
    my_code_end_line_index = my_code_start_line_index + 1

    for i in range(my_code_end_line_index):
        file_lines.append(file_helper_lines[i])

    file_lines.append(f'G1 F1200\n')
    for j in range(1, (layers_count + 1)):
        z = layer_height * j
        if j > 1:
            file_lines.append(f'G0 Z{z}\n')

        file_lines.append(f';LAYER_CHANGE\n')
        file_lines.append(f';{z}\n')

        if len(infill_printing_path) > 0:
            file_lines.append(';TYPE:Internal infill\n')
            for continuous_path in infill_printing_path:
                for idx, point in enumerate(continuous_path):
                    point = (x_convert_to_cartesian(point[0], 0, width, width) / 2 + offset,
                             y_convert_to_cartesian(point[1], 0, height, height) / 2 + offset)

                    if idx == 0:
                        file_lines.append(f'G1 X{point[0]} Y{point[1]} F9000\n')
                        file_lines.append(f'G1 F1200\n')
                    else:
                        prev_point = continuous_path[idx - 1]
                        prev_point = (x_convert_to_cartesian(prev_point[0], 0, width, width) / 2 + offset,
                                      y_convert_to_cartesian(prev_point[1], 0, height, height) / 2 + offset)
                        dist = math.dist(point, prev_point)
                        E = (4 * layer_height * flow_modifier * nozzle_diameter * dist) / (
                                math.pi * filament_diameter * filament_diameter)
                        file_lines.append(f'G1 X{point[0]} Y{point[1]} E{E}\n')

        if len(border_printing_path) > 0:
            file_lines.append(';TYPE:Perimeter\n')
            for continuous_path in border_printing_path:
                for idx, point in enumerate(continuous_path):
                    point = (x_convert_to_cartesian(point[0], 0, width, width) / 2 + offset,
                             y_convert_to_cartesian(point[1], 0, height, height) / 2 + offset)

                    if idx == 0:
                        file_lines.append(f'G1 X{point[0]} Y{point[1]} F9000\n')
                        file_lines.append(f'G1 F1200\n')
                    else:
                        prev_point = continuous_path[idx - 1]
                        prev_point = (x_convert_to_cartesian(prev_point[0], 0, width, width) / 2 + offset,
                                      y_convert_to_cartesian(prev_point[1], 0, height, height) / 2 + offset)
                        dist = math.dist(point, prev_point)
                        E = (4 * layer_height * flow_modifier * 4 * nozzle_diameter * dist) / (
                                math.pi * filament_diameter * filament_diameter)
                        file_lines.append(f'G1 X{point[0]} Y{point[1]} E{E}\n')

    for i in range(my_code_end_line_index, len(file_helper_lines)):
        file_lines.append(file_helper_lines[i])

    with open(f'{file_name}_LC_{layers_count}_FW_{flow_modifier}_.gcode', 'w', encoding='utf-8') as f:
        for line in file_lines:
            f.write(line)


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


def draw_trajectories_for_cells(screen, color_values_map, bitmap, cell_size=5):
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
                coefficient = 1
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
                    pygame.draw.line(screen, color, from_, to, 1)


def test_for_get_printing_path(screen, printing_path):
    for continuous_path in printing_path:
        for i in range(len(continuous_path) - 1):
            from_ = continuous_path[i]
            to = continuous_path[i + 1]
            pygame.draw.line(screen, (0, 0, 0), from_, to, 1)


def test_for_get_extended_bitmap(screen, extended_bitmap, cell_size=5):
    for i in range(len(extended_bitmap)):
        for j in range(len(extended_bitmap)):
            if extended_bitmap[i][j] != 0:
                for k in range(cell_size):
                    for m in range(cell_size):
                        if extended_bitmap[i][j][k][m] == 1:
                            top = i * cell_size + k
                            left = j * cell_size + m
                            pygame.draw.rect(screen, (0, 0, 0), (left, top, 1, 1))


def tests():
    # Pygame Settings
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('Tests')
    screen.fill((255, 255, 255))

    # Common part
    infill_img_path = 'images/8_infill.jpg'
    infill_img = cv2.imread(infill_img_path, cv2.IMREAD_GRAYSCALE)
    border_img_path = 'images/8_border.jpg'
    border_img = cv2.imread(border_img_path, cv2.IMREAD_GRAYSCALE)
    height, width = infill_img.shape
    cell_size = 5

    # img = pygame.image.load(infill_img_path)
    # screen.blit(img, (0, 0))

    # Infill
    threshold = 254
    color_values_map = get_color_values_map(infill_img, cell_size)
    bitmap = get_bitmap(color_values_map, threshold)
    printing_path = get_printing_path(bitmap, color_values_map)

    # Infill Tests
    # draw_trajectories_for_cells(screen, color_values_map, bitmap, cell_size)
    # test_for_get_color_values_map(screen, color_values_map, cell_size)
    # test_for_get_bitmap(screen, bitmap, cell_size)
    test_for_get_printing_path(screen, printing_path)

    # Border
    border_bitmap = get_bitmap_for_border(border_img, cell_size)
    extended_bitmap = get_extended_bitmap(border_img, border_bitmap, cell_size)
    border_path_for_traversing_grid = get_path_for_traversing_grid(border_bitmap)
    border_printing_path = get_printing_path(border_bitmap)

    # Border Tests
    # test_for_get_extended_bitmap(screen, extended_bitmap, cell_size)
    test_for_get_printing_path(screen, border_printing_path)

    # GCODE
    file_name = re.search(r'\d+_', infill_img_path).group(0)
    perimeter_type = 1
    print_options = {
        'layer_height': 0.2,
        'flow_modifier': 1,
        'nozzle_diameter': 0.4,
        'filament_diameter': 1.75,
        'offset': 20,
        'layers_count': 5,
        'width': width,
        'height': height,
        'file_name': f'results/new_img_{file_name}{"thick" if perimeter_type == 1 else "thin"}',
        'file_helper_name': 'helper_disk_2.gcode'
    }
    get_gcode_file(print_options, printing_path, border_printing_path)

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
