import csv
import math
import sys
import cv2
import pygame
import re

import standard_library_of_paths


# Utils
def get_vector_size(vector):
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])


def x_convert_to_cartesian(x, x_min, x_max, width):
    return x_min + ((x_max - x_min) * x / width)


def y_convert_to_cartesian(y, y_min, y_max, height):
    return y_max - ((y_max - y_min) * y / height)


def convert_pixel_points_to_cartesian(pixel_points, width, height, offset):
    points = []
    for pixel_point in pixel_points:
        x, y = pixel_point[0], pixel_point[1]
        point = (x_convert_to_cartesian(x, 0, width, width) / 2 + offset,
                 y_convert_to_cartesian(y, 0, height, height) / 2 + offset)
        points.append(point)

    return points


# Infill Part
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
    if first_segment[0] != second_segment[0]:
        if abs(second_segment[2] - first_segment[2]) < abs(second_segment[2] - first_segment[1]):
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
            coefficient = 0
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


def get_infill_points_for_printing(infill_printing_path, print_options):
    infill_points = []

    offset = print_options['offset']
    width = print_options['width']
    height = print_options['height']

    for continuous_path in infill_printing_path:
        for idx, point in enumerate(continuous_path):
            point = (x_convert_to_cartesian(point[0], 0, width, width) / 2 + offset,
                     y_convert_to_cartesian(point[1], 0, height, height) / 2 + offset)
            action_type = 'extrude'
            is_point_near_border = False

            if idx == 0:
                action_type = 'move'
            if (idx == 0) or (idx == (len(continuous_path) - 1)):
                is_point_near_border = True

            infill_points.append((point, action_type, is_point_near_border))

    return infill_points


# Border Part
def get_border_sides(border_points):
    border_sides = []
    for i in range(len(border_points) - 1):
        point_1 = border_points[i]
        point_2 = border_points[i + 1]
        border_sides.append((point_1, point_2))

    return border_sides


def get_border_sides_vectors(border_sides):
    border_sides_vectors = []
    for border_side in border_sides:
        point_1, point_2 = border_side[0], border_side[1]
        border_side_vector = (point_2[0] - point_1[0], point_2[1] - point_1[1])
        border_sides_vectors.append(border_side_vector)

    return border_sides_vectors


def get_border_sides_normals(border_sides_vectors):
    normals = []

    for border_side_vector in border_sides_vectors:
        x = border_side_vector[0]
        y = border_side_vector[1]
        normal = (y, -x)
        normals.append(normal)

    return normals


def shift_border_sides_by_normal(border_sides, border_sides_vectors, shift_coefficient):
    shifted_border_sides = []

    normals = get_border_sides_normals(border_sides_vectors)

    for i in range(0, len(border_sides)):
        border_side = border_sides[i]
        point_1, point_2 = border_side
        normal = normals[i]

        shifted_point_1 = (point_1[0] + shift_coefficient * normal[0] / get_vector_size(normal),
                           point_1[1] + shift_coefficient * normal[1] / get_vector_size(normal))
        shifted_point_2 = (point_2[0] + shift_coefficient * normal[0] / get_vector_size(normal),
                           point_2[1] + shift_coefficient * normal[1] / get_vector_size(normal))

        shifted_border_sides.append((shifted_point_1, shifted_point_2))

    return shifted_border_sides


def shit_border_points_by_normal(border_points, shift_coefficient):
    shifted_border_points = []
    border_sides = get_border_sides(border_points)
    border_sides_vectors = get_border_sides_vectors(border_sides)

    cross_product_sum = 0
    for i in range(len(border_sides_vectors) - 1):
        side_vector = border_sides_vectors[i]
        next_side_vector = border_sides_vectors[i + 1]
        cross_product_sum += side_vector[0] * next_side_vector[1] - side_vector[1] * next_side_vector[0]

    orientation = 1 if cross_product_sum >= 0 else -1
    shift_coefficient = shift_coefficient * orientation

    shifted_border_sides = shift_border_sides_by_normal(border_sides, border_sides_vectors, shift_coefficient)
    for shifted_border_side in shifted_border_sides:
        point_1, point_2 = shifted_border_side
        shifted_border_points.append(point_1)
        shifted_border_points.append(point_2)

    return shifted_border_points


# G-code
def get_gcode_file(print_options, infill_printing_path=(), border_points=()):
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

    if len(infill_printing_path) > 0 and len(border_points) > 0:
        file_name = file_name + '_total'
    elif len(infill_printing_path) > 0:
        file_name = file_name + '_infill'
    elif len(border_points) > 0:
        file_name = file_name + '_border'

    # Helper File
    with open(file_helper_name) as f:
        file_helper_lines = f.readlines()
    my_code_start_line_index = file_helper_lines.index('; my code start\n')
    my_code_end_line_index = my_code_start_line_index + 1
    for i in range(my_code_end_line_index):
        file_lines.append(file_helper_lines[i])

    # Prepare points for printing
    border_points = convert_pixel_points_to_cartesian(border_points, width, height, offset)
    shifted_border_points = shit_border_points_by_normal(border_points, print_options['nozzle_diameter'])
    infill_points = get_infill_points_for_printing(infill_printing_path, print_options)

    # Get G-code
    file_lines.append(f'G1 F1200\n')
    for j in range(1, (layers_count + 1)):
        z = layer_height * j
        file_lines.append(f'G0 Z{z}\n')
        file_lines.append(f';LAYER_CHANGE\n')
        file_lines.append(f';{z}\n')

        # Outer Border
        if len(shifted_border_points) > 0:
            file_lines.append(';TYPE:External perimeter\n')

            for idx, point in enumerate(shifted_border_points):
                if idx == 0:
                    file_lines.append(f'G1 X{point[0]} Y{point[1]} F9000\n')
                    file_lines.append(f'G1 F1200\n')
                else:
                    prev_point = shifted_border_points[idx - 1]
                    dist = math.dist(point, prev_point)
                    E = (4 * layer_height * flow_modifier * nozzle_diameter * dist) / (
                            math.pi * filament_diameter * filament_diameter)
                    file_lines.append(f'G1 X{point[0]} Y{point[1]} E{E}\n')

            first_border_point = shifted_border_points[0]
            last_border_point = shifted_border_points[len(shifted_border_points) - 1]
            dist = math.dist(first_border_point, last_border_point)
            E = (4 * layer_height * flow_modifier * nozzle_diameter * dist) / (
                    math.pi * filament_diameter * filament_diameter)
            file_lines.append(f'G1 X{first_border_point[0]} Y{first_border_point[1]} E{E}\n')

        # Border
        if len(border_points) > 0:
            file_lines.append(';TYPE:Perimeter\n')

            for idx, point in enumerate(border_points):
                if idx == 0:
                    file_lines.append(f'G1 X{point[0]} Y{point[1]} F9000\n')
                    file_lines.append(f'G1 F1200\n')
                else:
                    prev_point = border_points[idx - 1]
                    dist = math.dist(point, prev_point)
                    E = (4 * layer_height * flow_modifier * nozzle_diameter * dist) / (
                            math.pi * filament_diameter * filament_diameter)
                    file_lines.append(f'G1 X{point[0]} Y{point[1]} E{E}\n')

            first_border_point = border_points[0]
            last_border_point = border_points[len(border_points) - 1]
            dist = math.dist(first_border_point, last_border_point)
            E = (4 * layer_height * flow_modifier * nozzle_diameter * dist) / (
                    math.pi * filament_diameter * filament_diameter)
            file_lines.append(f'G1 X{first_border_point[0]} Y{first_border_point[1]} E{E}\n')

        # Infill
        if len(infill_points) > 0:
            file_lines.append(';TYPE:Internal infill\n')
            for i in range(len(infill_points)):
                point, action_type, is_point_near_border = infill_points[i]

                if is_point_near_border and (len(border_points) > 0):
                    new_point = border_points[0]
                    dist = math.dist(point, new_point)
                    for border_point in border_points:
                        temp_dist = math.dist(point, border_point)
                        if temp_dist < dist:
                            dist = temp_dist
                            new_point = border_point

                    point = new_point

                if action_type == 'move':
                    file_lines.append(f'G1 X{point[0]} Y{point[1]} F9000\n')
                    file_lines.append(f'G1 F1200\n')
                else:
                    prev_point, _, _ = infill_points[i - 1]
                    dist = math.dist(point, prev_point)
                    E = (4 * layer_height * flow_modifier * nozzle_diameter * dist) / (
                            math.pi * filament_diameter * filament_diameter)
                    file_lines.append(f'G1 X{point[0]} Y{point[1]} E{E}\n')

    # Helper File
    for i in range(my_code_end_line_index, len(file_helper_lines)):
        file_lines.append(file_helper_lines[i])

    # Result File
    with open(f'{file_name}_LC{layers_count}_FW{flow_modifier}.gcode', 'w', encoding='utf-8') as f:
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


def tests():
    # Pygame Settings
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('Tests')
    screen.fill((255, 255, 255))

    # Common part
    infill_img_path = 'images/13.jpg'
    infill_img = cv2.imread(infill_img_path, cv2.IMREAD_GRAYSCALE)
    border_img_path = 'images/13.csv'
    height, width = infill_img.shape
    cell_size = 5

    # img = pygame.image.load(infill_img_path)
    # screen.blit(img, (0, 0))

    # Infill
    threshold = 254
    color_values_map = get_color_values_map(infill_img, cell_size)
    bitmap = get_bitmap(color_values_map, threshold)
    infill_printing_path = get_printing_path(bitmap, color_values_map, cell_size)

    # Infill Tests
    # draw_trajectories_for_cells(screen, color_values_map, bitmap, cell_size)
    # test_for_get_color_values_map(screen, color_values_map, cell_size)
    # test_for_get_bitmap(screen, bitmap, cell_size)
    # test_for_get_printing_path(screen, infill_printing_path)

    # Border
    with open(border_img_path) as f:
        svg_border_points = list(csv.reader(f, delimiter=','))[1:]

    # Close the contour
    svg_border_points.append(svg_border_points[0])

    # Convert to float
    for i in range(len(svg_border_points)):
        point = svg_border_points[i]
        svg_border_points[i] = (float(point[0]), float(point[1]))

    # GCODE
    file_name = ''
    file_name = re.search(r'\d+_?(small|\d+)?', infill_img_path).group(0)
    print_options = {
        'layer_height': 0.2,
        'flow_modifier': 1,
        'nozzle_diameter': 0.4,
        'filament_diameter': 1.75,
        'offset': 20,
        'layers_count': 10,
        'width': width,
        'height': height,
        'file_name': f'results/img_{file_name}',
        'file_helper_name': 'helper_disk_2.gcode'
    }
    get_gcode_file(print_options, infill_printing_path, svg_border_points)

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
