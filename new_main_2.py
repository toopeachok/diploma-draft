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


def convert_pixel_points_to_cartesian(pixel_points, width, height, offset):
    points = []
    for pixel_point in pixel_points:
        point = (x_convert_to_cartesian(pixel_point[0], 0, width, width) / 2 + offset,
                 y_convert_to_cartesian(pixel_point[1], 0, height, height) / 2 + offset)
        points.append(point)

    return points


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

    with open(file_helper_name) as f:
        file_helper_lines = f.readlines()

    my_code_start_line_index = file_helper_lines.index('; my code start\n')
    my_code_end_line_index = my_code_start_line_index + 1

    for i in range(my_code_end_line_index):
        file_lines.append(file_helper_lines[i])

    border_points = convert_pixel_points_to_cartesian(border_points, width, height, offset)
    infill_points = []

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

    file_lines.append(f'G1 F1200\n')
    for j in range(1, (layers_count + 1)):
        z = layer_height * j
        if j > 1:
            file_lines.append(f'G0 Z{z}\n')

        file_lines.append(f';LAYER_CHANGE\n')
        file_lines.append(f';{z}\n')

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
    infill_img_path = 'images/6_infill.jpg'
    infill_img = cv2.imread(infill_img_path, cv2.IMREAD_GRAYSCALE)
    border_img_path = 'images/6_border.jpg'
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
    # test_for_get_printing_path(screen, printing_path)

    # Border
    border_bitmap = get_bitmap_for_border(border_img, cell_size)
    extended_bitmap = get_extended_bitmap(border_img, border_bitmap, cell_size)
    border_path_for_traversing_grid = get_path_for_traversing_grid(border_bitmap)
    border_printing_path = get_printing_path(border_bitmap)
    svg_border_points = [[151.07400512695312, 23.95330047607422], [150.52102661132812, 23.385387420654297],
                         [149.96803283691406, 22.88312530517578], [149.41505432128906, 23.45103645324707],
                         [148.862060546875, 24.018932342529297], [148.30894470214844, 24.586727142333984],
                         [147.75584411621094, 25.154518127441406], [147.20272827148438, 25.722312927246094],
                         [146.64962768554688, 26.29010772705078], [146.09652709960938, 26.857898712158203],
                         [145.5434112548828, 27.42569351196289], [144.9903106689453, 27.993484497070312],
                         [144.4372100830078, 28.561279296875], [143.88409423828125, 29.129074096679688],
                         [143.33099365234375, 29.69686508178711], [142.77789306640625, 30.264659881591797],
                         [142.2247772216797, 30.83245086669922], [141.6716766357422, 31.400245666503906],
                         [141.11856079101562, 31.968040466308594], [140.56546020507812, 32.535831451416016],
                         [140.01235961914062, 33.10362243652344], [139.45925903320312, 33.67142105102539],
                         [138.90614318847656, 34.23921203613281], [138.35304260253906, 34.8070068359375],
                         [137.7999267578125, 35.37479782104492], [137.246826171875, 35.942588806152344],
                         [136.6937255859375, 36.51038360595703], [136.140625, 37.07817459106445],
                         [135.58750915527344, 37.64596939086914], [135.03440856933594, 38.21376037597656],
                         [134.48129272460938, 38.78155517578125], [133.92819213867188, 39.34934997558594],
                         [133.37509155273438, 39.917144775390625], [132.8219757080078, 40.48493576049805],
                         [132.2688751220703, 41.05272674560547], [131.71575927734375, 41.620521545410156],
                         [131.16265869140625, 42.188316345214844], [130.60955810546875, 42.75611114501953],
                         [130.05645751953125, 43.32390213012695], [129.5033416748047, 43.891693115234375],
                         [128.9502410888672, 44.45948791503906], [128.39712524414062, 45.02728271484375],
                         [127.84402465820312, 45.59507751464844], [127.2909164428711, 46.16286849975586],
                         [126.73780822753906, 46.73065948486328], [126.18470764160156, 47.29845428466797],
                         [125.63159942626953, 47.866249084472656], [125.07849884033203, 48.43404006958008],
                         [124.525390625, 49.0018310546875], [123.97228240966797, 49.56962585449219],
                         [123.41918182373047, 50.137420654296875], [122.86607360839844, 50.7052116394043],
                         [122.3129653930664, 51.273006439208984], [121.75985717773438, 51.840797424316406],
                         [121.20675659179688, 52.408592224121094], [120.65364837646484, 52.97638702392578],
                         [120.10054016113281, 53.5441780090332], [119.54743957519531, 54.11197280883789],
                         [118.99433135986328, 54.67976379394531], [118.44122314453125, 55.24755859375],
                         [117.88812255859375, 55.81535339355469], [117.33501434326172, 56.38314437866211],
                         [116.78190612792969, 56.9509391784668], [116.22879791259766, 57.51873016357422],
                         [115.67569732666016, 58.086524963378906], [115.12258911132812, 58.654319763183594],
                         [114.5694808959961, 59.222110748291016], [114.01637268066406, 59.7899055480957],
                         [113.46327209472656, 60.357696533203125], [112.91016387939453, 60.92549133300781],
                         [112.3570556640625, 61.4932861328125], [111.803955078125, 62.06107711791992],
                         [111.25084686279297, 62.62887191772461], [110.69773864746094, 63.19666290283203],
                         [110.1446304321289, 63.76445770263672], [109.5915298461914, 64.3322525024414],
                         [109.03842163085938, 64.9000473022461], [108.48531341552734, 65.46783447265625],
                         [107.93220520019531, 66.03562927246094], [107.37910461425781, 66.60342407226562],
                         [106.82599639892578, 67.17121887207031], [106.27288818359375, 67.739013671875],
                         [105.71978759765625, 68.30680084228516], [105.16667938232422, 68.87459564208984],
                         [104.61357116699219, 69.44239044189453], [104.06046295166016, 70.01018524169922],
                         [103.35150146484375, 70.30802154541016], [102.58322143554688, 70.50312805175781],
                         [101.81494903564453, 70.69822692871094], [101.04667663574219, 70.89332580566406],
                         [100.27839660644531, 71.08843231201172], [99.51011657714844, 71.28353118896484],
                         [98.74183654785156, 71.4786376953125], [97.97355651855469, 71.67373657226562],
                         [97.20529174804688, 71.86883544921875], [96.43701171875, 72.06393432617188],
                         [95.66873168945312, 72.25904083251953], [94.90045166015625, 72.45413970947266],
                         [94.1321792602539, 72.64924621582031], [93.36389923095703, 72.84434509277344],
                         [92.59562683105469, 73.03944396972656], [91.82734680175781, 73.23455047607422],
                         [91.05906677246094, 73.42964935302734], [90.29078674316406, 73.62474822998047],
                         [89.52250671386719, 73.81985473632812], [88.75423431396484, 74.01495361328125],
                         [87.98595428466797, 74.21005249023438], [87.21768188476562, 74.40515899658203],
                         [86.44940185546875, 74.60025787353516], [85.68112182617188, 74.79536437988281],
                         [84.912841796875, 74.99046325683594], [84.14456176757812, 75.18556213378906],
                         [83.37628936767578, 75.38066864013672], [82.6080093383789, 75.57576751708984],
                         [81.83973693847656, 75.77086639404297], [81.07145690917969, 75.96597290039062],
                         [80.30317687988281, 76.16107177734375], [79.53489685058594, 76.35617065429688],
                         [78.76661682128906, 76.55127716064453], [77.99834442138672, 76.74637603759766],
                         [77.23006439208984, 76.94148254394531], [76.4617919921875, 77.13658142089844],
                         [75.69351196289062, 77.33168029785156], [74.92523193359375, 77.52678680419922],
                         [74.15695190429688, 77.72188568115234], [73.388671875, 77.91698455810547],
                         [72.62039947509766, 78.11209106445312], [71.85211944580078, 78.30718994140625],
                         [71.0838394165039, 78.5022964477539], [70.31556701660156, 78.69739532470703],
                         [69.54728698730469, 78.89249420166016], [68.77900695800781, 79.08760070800781],
                         [68.01072692871094, 79.28269958496094], [67.2424545288086, 79.47779846191406],
                         [66.47417449951172, 79.67290496826172], [65.70589447021484, 79.86800384521484],
                         [64.9376220703125, 80.0631103515625], [64.16934204101562, 80.25820922851562],
                         [63.40106201171875, 80.45330810546875], [62.63278579711914, 80.6484146118164],
                         [61.86450958251953, 80.84351348876953], [61.096229553222656, 81.03861236572266],
                         [60.32795333862305, 81.23371887207031], [59.55967712402344, 81.42881774902344],
                         [58.79139709472656, 81.62391662597656], [58.02311706542969, 81.81902313232422],
                         [57.25484085083008, 82.01412200927734], [56.48656463623047, 82.209228515625],
                         [55.718284606933594, 82.40432739257812], [54.950008392333984, 82.59942626953125],
                         [54.18172836303711, 82.7945327758789], [53.4134521484375, 82.98963165283203],
                         [52.645172119140625, 83.18473052978516], [51.876895904541016, 83.37983703613281],
                         [51.108619689941406, 83.57493591308594], [50.34033966064453, 83.77003479003906],
                         [49.57206344604492, 83.96514129638672], [48.80379104614258, 84.16024017333984],
                         [48.03551483154297, 84.35533905029297], [47.267234802246094, 84.55044555664062],
                         [46.498958587646484, 84.74554443359375], [45.730682373046875, 84.94064331054688],
                         [44.96239471435547, 85.13574981689453], [44.19411849975586, 85.33084869384766],
                         [43.425838470458984, 85.52595520019531], [42.657562255859375, 85.72105407714844],
                         [41.8892822265625, 85.91615295410156], [41.12101745605469, 86.11127471923828],
                         [40.35274887084961, 86.3064193725586], [39.96929931640625, 86.72371673583984],
                         [40.18447494506836, 87.48661804199219], [40.39965057373047, 88.24951171875],
                         [40.614830017089844, 89.01241302490234], [40.83000946044922, 89.77531433105469],
                         [41.04518127441406, 90.5382080078125], [41.26036071777344, 91.30110931396484],
                         [41.47554016113281, 92.06401062011719], [41.69071578979492, 92.826904296875],
                         [41.90589141845703, 93.58980560302734], [42.121070861816406, 94.35270690917969],
                         [42.336246490478516, 95.1156005859375], [42.551422119140625, 95.87850189208984],
                         [42.7666015625, 96.64140319824219], [42.98177719116211, 97.404296875],
                         [43.19695281982422, 98.16719818115234], [43.412132263183594, 98.93009948730469],
                         [43.6273078918457, 99.6929931640625], [43.84248733520508, 100.45589447021484],
                         [44.05765914916992, 101.21878051757812], [44.27283477783203, 101.98167419433594],
                         [44.488014221191406, 102.74457550048828], [44.703189849853516, 103.50747680664062],
                         [44.918365478515625, 104.27037048339844], [45.133544921875, 105.03327178955078],
                         [45.34872055053711, 105.79617309570312], [45.563899993896484, 106.55906677246094],
                         [45.779075622558594, 107.32196807861328], [45.99425506591797, 108.08486938476562],
                         [46.20943069458008, 108.84776306152344], [46.42460632324219, 109.61066436767578],
                         [46.63978576660156, 110.37356567382812], [46.85496139526367, 111.13645935058594],
                         [47.07013702392578, 111.89936065673828], [47.285316467285156, 112.66226196289062],
                         [47.500492095947266, 113.42515563964844], [47.715667724609375, 114.18805694580078],
                         [47.93084716796875, 114.95095825195312], [48.146026611328125, 115.71385192871094],
                         [48.361202239990234, 116.47675323486328], [48.576377868652344, 117.23965454101562],
                         [48.79155731201172, 118.00254821777344], [49.00673294067383, 118.76544952392578],
                         [49.22190856933594, 119.52835083007812], [49.43708801269531, 120.29124450683594],
                         [49.65226364135742, 121.05414581298828], [49.86743927001953, 121.8170394897461],
                         [50.082618713378906, 122.57994079589844], [50.297794342041016, 123.34284210205078],
                         [50.51297378540039, 124.1057357788086], [50.7281494140625, 124.86863708496094],
                         [50.943328857421875, 125.63153076171875], [51.158504486083984, 126.3944320678711],
                         [51.373680114746094, 127.15733337402344], [51.58885955810547, 127.92022705078125],
                         [51.80403518676758, 128.68313598632812], [52.01921081542969, 129.44602966308594],
                         [52.23439025878906, 130.20892333984375], [52.44956588745117, 130.97183227539062],
                         [52.66474533081055, 131.73472595214844], [52.879920959472656, 132.49761962890625],
                         [53.095096588134766, 133.26052856445312], [53.31027603149414, 134.02342224121094],
                         [53.52545166015625, 134.78631591796875], [53.740631103515625, 135.54922485351562],
                         [53.955806732177734, 136.31211853027344], [54.170982360839844, 137.07501220703125],
                         [54.38616180419922, 137.83792114257812], [54.60133743286133, 138.60081481933594],
                         [54.81651306152344, 139.36370849609375], [55.03169250488281, 140.12661743164062],
                         [55.24686813354492, 140.88951110839844], [55.4620475769043, 141.65240478515625],
                         [55.677223205566406, 142.41531372070312], [55.892398834228516, 143.17820739746094],
                         [56.10757827758789, 143.94110107421875], [56.32275390625, 144.70399475097656],
                         [56.537933349609375, 145.46690368652344], [56.753108978271484, 146.22979736328125],
                         [56.968284606933594, 146.99269104003906], [57.18346405029297, 147.75559997558594],
                         [57.39863967895508, 148.51849365234375], [57.61381912231445, 149.28138732910156],
                         [57.80400848388672, 150.04428100585938], [57.588836669921875, 150.8071746826172],
                         [57.3736572265625, 151.570068359375], [57.15848159790039, 152.33297729492188],
                         [56.94330596923828, 153.0958709716797], [56.728126525878906, 153.8587646484375],
                         [56.51295471191406, 154.62167358398438], [56.29777526855469, 155.3845672607422],
                         [56.08259963989258, 156.1474609375], [55.86742401123047, 156.91036987304688],
                         [55.652244567871094, 157.6732635498047], [55.437068939208984, 158.4361572265625],
                         [55.221893310546875, 159.19906616210938], [55.006717681884766, 159.9619598388672],
                         [54.791542053222656, 160.724853515625], [54.57636260986328, 161.48776245117188],
                         [54.36118698120117, 162.2506561279297], [54.14601135253906, 163.0135498046875],
                         [53.93083572387695, 163.77645874023438], [53.715660095214844, 164.5393524169922],
                         [53.50048065185547, 165.30224609375], [53.28530502319336, 166.06515502929688],
                         [53.07012939453125, 166.8280487060547], [52.85495376586914, 167.5909423828125],
                         [52.63977813720703, 168.35385131835938], [52.424598693847656, 169.1167449951172],
                         [52.20942306518555, 169.879638671875], [51.99424743652344, 170.64254760742188],
                         [51.77906799316406, 171.4054412841797], [51.56389617919922, 172.16835021972656],
                         [51.348716735839844, 172.93124389648438], [51.133541107177734, 173.6941375732422],
                         [50.918365478515625, 174.45704650878906], [50.70318603515625, 175.21994018554688],
                         [50.48801040649414, 175.9828338623047], [50.27283477783203, 176.74574279785156],
                         [50.05765914916992, 177.50863647460938], [49.84248352050781, 178.2715301513672],
                         [49.62730407714844, 179.03443908691406], [49.41212844848633, 179.79733276367188],
                         [49.19695281982422, 180.5602264404297], [48.98177719116211, 181.32313537597656],
                         [48.7666015625, 182.08602905273438], [48.551422119140625, 182.8489227294922],
                         [48.336246490478516, 183.61183166503906], [48.121070861816406, 184.37472534179688],
                         [47.9058952331543, 185.1376190185547], [47.69071960449219, 185.90052795410156],
                         [47.47554016113281, 186.66342163085938], [47.2603645324707, 187.4263153076172],
                         [47.045188903808594, 188.18922424316406], [46.830013275146484, 188.95211791992188],
                         [46.61483383178711, 189.7150115966797], [46.399658203125, 190.47792053222656],
                         [46.18448257446289, 191.24081420898438], [45.96930694580078, 192.00372314453125],
                         [45.75413131713867, 192.76661682128906], [45.5389518737793, 193.52951049804688],
                         [45.32377624511719, 194.29241943359375], [45.10860061645508, 195.05531311035156],
                         [44.89342498779297, 195.81820678710938], [44.678253173828125, 196.58108520507812],
                         [44.46307373046875, 197.343994140625], [44.24789810180664, 198.1068878173828],
                         [44.03272247314453, 198.86978149414062], [43.81754684448242, 199.6326904296875],
                         [43.60237121582031, 200.3955841064453], [43.38719177246094, 201.1584930419922],
                         [43.17201614379883, 201.92138671875], [42.95684051513672, 202.6842803955078],
                         [42.74166488647461, 203.4471893310547], [42.52648162841797, 204.21009826660156],
                         [42.31130599975586, 204.97299194335938], [42.09613037109375, 205.73590087890625],
                         [41.88095474243164, 206.49879455566406], [41.665775299072266, 207.26168823242188],
                         [41.450599670410156, 208.02459716796875], [41.23542404174805, 208.78749084472656],
                         [41.02024841308594, 209.55038452148438], [40.80507278442383, 210.31329345703125],
                         [40.58989334106445, 211.07618713378906], [40.374717712402344, 211.83908081054688],
                         [40.1595458984375, 212.6019744873047], [39.944366455078125, 213.3648681640625],
                         [40.44178771972656, 213.7163543701172], [41.21006774902344, 213.9114532470703],
                         [41.97835922241211, 214.10655212402344], [42.746639251708984, 214.30165100097656],
                         [43.51491928100586, 214.4967498779297], [44.28319549560547, 214.6918487548828],
                         [45.051475524902344, 214.88693237304688], [45.81975555419922, 215.08203125],
                         [46.588035583496094, 215.27713012695312], [47.3563117980957, 215.47222900390625],
                         [48.12459182739258, 215.66732788085938], [48.89287185668945, 215.8624267578125],
                         [49.66115188598633, 216.05752563476562], [50.42942810058594, 216.25262451171875],
                         [51.19770812988281, 216.44772338867188], [51.96598815917969, 216.642822265625],
                         [52.7342643737793, 216.83790588378906], [53.50254440307617, 217.0330047607422],
                         [54.27082443237305, 217.2281036376953], [55.03910446166992, 217.42320251464844],
                         [55.80738067626953, 217.61830139160156], [56.575660705566406, 217.8134002685547],
                         [57.34394073486328, 218.0084991455078], [58.112220764160156, 218.20359802246094],
                         [58.880496978759766, 218.398681640625], [59.64877700805664, 218.59378051757812],
                         [60.417057037353516, 218.78887939453125], [61.18533706665039, 218.98397827148438],
                         [61.95361328125, 219.1790771484375], [62.721893310546875, 219.37417602539062],
                         [63.49017333984375, 219.56927490234375], [64.25845336914062, 219.76437377929688],
                         [65.0267333984375, 219.95947265625], [65.79501342773438, 220.15455627441406],
                         [66.56329345703125, 220.3496551513672], [67.3315658569336, 220.5447540283203],
                         [68.09984588623047, 220.73985290527344], [68.86812591552734, 220.93495178222656],
                         [69.63640594482422, 221.1300506591797], [70.40465545654297, 221.32513427734375],
                         [71.17292785644531, 221.52023315429688], [71.94120788574219, 221.71533203125],
                         [72.70948791503906, 221.91043090820312], [73.47776794433594, 222.10552978515625],
                         [74.24604797363281, 222.30062866210938], [75.01432800292969, 222.4957275390625],
                         [75.78260803222656, 222.69081115722656], [76.55088806152344, 222.8859100341797],
                         [77.31916809082031, 223.0810089111328], [78.08744812011719, 223.27610778808594],
                         [78.85572814941406, 223.47120666503906], [79.6240005493164, 223.6663055419922],
                         [80.39228057861328, 223.8614044189453], [81.16056060791016, 224.05650329589844],
                         [81.92884063720703, 224.25160217285156], [82.6971206665039, 224.44668579101562],
                         [83.46539306640625, 224.64178466796875], [84.23367309570312, 224.83688354492188],
                         [85.001953125, 225.031982421875], [85.77023315429688, 225.22708129882812],
                         [86.53851318359375, 225.42218017578125], [87.30679321289062, 225.61727905273438],
                         [88.0750732421875, 225.8123779296875], [88.84335327148438, 226.00747680664062],
                         [89.61163330078125, 226.2025604248047], [90.3799057006836, 226.3976593017578],
                         [91.14818572998047, 226.59275817871094], [91.91646575927734, 226.78785705566406],
                         [92.68474578857422, 226.9829559326172], [93.4530258178711, 227.1780548095703],
                         [94.22130584716797, 227.37315368652344], [94.98958587646484, 227.56825256347656],
                         [95.75785827636719, 227.76333618164062], [96.52613830566406, 227.95843505859375],
                         [97.29441833496094, 228.15353393554688], [98.06269836425781, 228.3486328125],
                         [98.83097839355469, 228.54373168945312], [99.59925842285156, 228.73883056640625],
                         [100.36753845214844, 228.93392944335938], [101.13581848144531, 229.1290283203125],
                         [101.90409088134766, 229.32412719726562], [102.67237091064453, 229.5192108154297],
                         [103.4406509399414, 229.7143096923828], [104.1246566772461, 230.05540466308594],
                         [104.6777572631836, 230.62319946289062], [105.23086547851562, 231.19100952148438],
                         [105.78396606445312, 231.75880432128906], [106.33706665039062, 232.32659912109375],
                         [106.89016723632812, 232.89439392089844], [107.44326782226562, 233.46218872070312],
                         [107.99636840820312, 234.0299835205078], [108.54946899414062, 234.5977783203125],
                         [109.10257720947266, 235.16558837890625], [109.65567779541016, 235.73338317871094],
                         [110.20878601074219, 236.30117797851562], [110.76188659667969, 236.8689727783203],
                         [111.31498718261719, 237.436767578125], [111.86808776855469, 238.0045623779297],
                         [112.42118835449219, 238.57235717773438], [112.97428894042969, 239.14016723632812],
                         [113.52739715576172, 239.70794677734375], [114.08049774169922, 240.2757568359375],
                         [114.63360595703125, 240.8435516357422], [115.18670654296875, 241.41134643554688],
                         [115.73980712890625, 241.97914123535156], [116.29290771484375, 242.54693603515625],
                         [116.84600830078125, 243.11474609375], [117.39910888671875, 243.68252563476562],
                         [117.95221710205078, 244.25033569335938], [118.50531768798828, 244.81813049316406],
                         [119.05841827392578, 245.38592529296875], [119.61152648925781, 245.95372009277344],
                         [120.16462707519531, 246.52151489257812], [120.71772766113281, 247.0893096923828],
                         [121.27082824707031, 247.6571044921875], [121.82392883300781, 248.22491455078125],
                         [122.37703704833984, 248.79270935058594], [122.93013763427734, 249.36050415039062],
                         [123.48323822021484, 249.9282989501953], [124.03634643554688, 250.49609375],
                         [124.58944702148438, 251.0638885498047], [125.14254760742188, 251.63168334960938],
                         [125.69564819335938, 252.19949340820312], [126.24874877929688, 252.7672882080078],
                         [126.8018569946289, 253.3350830078125], [127.3549575805664, 253.9028778076172],
                         [127.9080581665039, 254.47067260742188], [128.46116638183594, 255.03846740722656],
                         [129.01426696777344, 255.60626220703125], [129.56736755371094, 256.174072265625],
                         [130.12046813964844, 256.7418518066406], [130.67356872558594, 257.3096618652344],
                         [131.22666931152344, 257.87744140625], [131.77978515625, 258.44525146484375],
                         [132.3328857421875, 259.0130615234375], [132.885986328125, 259.5808410644531],
                         [133.4390869140625, 260.1486511230469], [133.9921875, 260.7164306640625],
                         [134.5452880859375, 261.28424072265625], [135.098388671875, 261.8520202636719],
                         [135.6514892578125, 262.4198303222656], [136.20458984375, 262.9876403808594],
                         [136.75770568847656, 263.555419921875], [137.31080627441406, 264.12322998046875],
                         [137.86390686035156, 264.6910095214844], [138.41700744628906, 265.2588195800781],
                         [138.97010803222656, 265.82659912109375], [139.52320861816406, 266.3944091796875],
                         [140.07630920410156, 266.96221923828125], [140.62940979003906, 267.5299987792969],
                         [141.18252563476562, 268.0978088378906], [141.73562622070312, 268.66558837890625],
                         [142.28872680664062, 269.2333984375], [142.84182739257812, 269.8011779785156],
                         [143.39492797851562, 270.3689880371094], [143.94802856445312, 270.936767578125],
                         [144.50112915039062, 271.50457763671875], [145.05422973632812, 272.0723876953125],
                         [145.60733032226562, 272.6401672363281], [146.1604461669922, 273.2079772949219],
                         [146.7135467529297, 273.7757568359375], [147.2666473388672, 274.34356689453125],
                         [147.8197479248047, 274.9113464355469], [148.3728485107422, 275.4791564941406],
                         [148.9259490966797, 276.0469665527344], [149.4789276123047, 276.6148681640625],
                         [150.0319061279297, 277.1172180175781], [150.5848846435547, 276.54931640625],
                         [151.1378936767578, 275.9814147949219], [151.6909942626953, 275.4136047363281],
                         [152.24411010742188, 274.8458251953125], [152.79721069335938, 274.27801513671875],
                         [153.35031127929688, 273.7102355957031], [153.90341186523438, 273.1424255371094],
                         [154.45651245117188, 272.57464599609375], [155.00961303710938, 272.0068359375],
                         [155.56271362304688, 271.43902587890625], [156.11581420898438, 270.8712463378906],
                         [156.66891479492188, 270.3034362792969], [157.22201538085938, 269.73565673828125],
                         [157.77511596679688, 269.1678466796875], [158.32823181152344, 268.60003662109375],
                         [158.88133239746094, 268.0322570800781], [159.43443298339844, 267.4644470214844],
                         [159.98753356933594, 266.89666748046875], [160.54063415527344, 266.328857421875],
                         [161.09373474121094, 265.7610778808594], [161.64683532714844, 265.1932678222656],
                         [162.199951171875, 264.62548828125], [162.7530517578125, 264.05767822265625],
                         [163.30615234375, 263.4898681640625], [163.8592529296875, 262.9220886230469],
                         [164.412353515625, 262.3542785644531], [164.9654541015625, 261.7864990234375],
                         [165.5185546875, 261.21868896484375], [166.0716552734375, 260.6509094238281],
                         [166.624755859375, 260.0830993652344], [167.1778564453125, 259.5152893066406],
                         [167.73097229003906, 258.947509765625], [168.28407287597656, 258.37969970703125],
                         [168.83717346191406, 257.8119201660156], [169.39027404785156, 257.2441101074219],
                         [169.94337463378906, 256.67633056640625], [170.49647521972656, 256.1085205078125],
                         [171.04957580566406, 255.5407257080078], [171.60269165039062, 254.97293090820312],
                         [172.15579223632812, 254.40512084960938], [172.70889282226562, 253.8373260498047],
                         [173.26199340820312, 253.26953125], [173.81509399414062, 252.7017364501953],
                         [174.36819458007812, 252.13394165039062], [174.92129516601562, 251.56614685058594],
                         [175.47439575195312, 250.99835205078125], [176.02749633789062, 250.43055725097656],
                         [176.58059692382812, 249.8627471923828], [177.1337127685547, 249.29495239257812],
                         [177.6868133544922, 248.72715759277344], [178.2399139404297, 248.15936279296875],
                         [178.7930145263672, 247.59156799316406], [179.3461151123047, 247.02377319335938],
                         [179.8992156982422, 246.4559783935547], [180.4523162841797, 245.88818359375],
                         [181.0054168701172, 245.32037353515625], [181.55853271484375, 244.75257873535156],
                         [182.11163330078125, 244.18478393554688], [182.66473388671875, 243.6169891357422],
                         [183.21783447265625, 243.0491943359375], [183.77093505859375, 242.4813995361328],
                         [184.32403564453125, 241.91360473632812], [184.87713623046875, 241.34579467773438],
                         [185.43023681640625, 240.7779998779297], [185.98333740234375, 240.210205078125],
                         [186.5364532470703, 239.6424102783203], [187.0895538330078, 239.07461547851562],
                         [187.6426544189453, 238.50682067871094], [188.1957550048828, 237.93902587890625],
                         [188.7488555908203, 237.3712158203125], [189.3019561767578, 236.8034210205078],
                         [189.8550567626953, 236.23562622070312], [190.4081573486328, 235.66783142089844],
                         [190.9612579345703, 235.10003662109375], [191.51437377929688, 234.53224182128906],
                         [192.06747436523438, 233.96444702148438], [192.62057495117188, 233.39663696289062],
                         [193.17367553710938, 232.82884216308594], [193.72677612304688, 232.26104736328125],
                         [194.27987670898438, 231.69325256347656], [194.83297729492188, 231.12545776367188],
                         [195.38607788085938, 230.5576629638672], [195.93919372558594, 229.9898681640625],
                         [196.6479949951172, 229.69180297851562], [197.41627502441406, 229.4967041015625],
                         [198.18455505371094, 229.30160522460938], [198.9528350830078, 229.10650634765625],
                         [199.7211151123047, 228.91140747070312], [200.48939514160156, 228.71630859375],
                         [201.25767517089844, 228.52120971679688], [202.02593994140625, 228.32611083984375],
                         [202.79421997070312, 228.13101196289062], [203.5625, 227.9359130859375],
                         [204.33078002929688, 227.74081420898438], [205.09906005859375, 227.54571533203125],
                         [205.86734008789062, 227.35061645507812], [206.6356201171875, 227.155517578125],
                         [207.40390014648438, 226.96041870117188], [208.17218017578125, 226.76531982421875],
                         [208.94046020507812, 226.57022094726562], [209.708740234375, 226.3751220703125],
                         [210.47702026367188, 226.18003845214844], [211.24530029296875, 225.9849395751953],
                         [212.01358032226562, 225.7898406982422], [212.78184509277344, 225.59474182128906],
                         [213.5501251220703, 225.39964294433594], [214.3184051513672, 225.2045440673828],
                         [215.08668518066406, 225.0094451904297], [215.85496520996094, 224.81434631347656],
                         [216.6232452392578, 224.61924743652344], [217.3915252685547, 224.4241485595703],
                         [218.15980529785156, 224.2290496826172], [218.92808532714844, 224.03395080566406],
                         [219.69635009765625, 223.83885192871094], [220.46463012695312, 223.6437530517578],
                         [221.23291015625, 223.4486541748047], [222.00119018554688, 223.25355529785156],
                         [222.76947021484375, 223.05845642089844], [223.53775024414062, 222.8633575439453],
                         [224.3060302734375, 222.6682586669922], [225.07431030273438, 222.47315979003906],
                         [225.84259033203125, 222.27806091308594], [226.61087036132812, 222.0829620361328],
                         [227.37911987304688, 221.88787841796875], [228.14739990234375, 221.69277954101562],
                         [228.91567993164062, 221.4976806640625], [229.6839599609375, 221.30258178710938],
                         [230.45223999023438, 221.10748291015625], [231.2205047607422, 220.91238403320312],
                         [231.98878479003906, 220.71728515625], [232.75706481933594, 220.52218627929688],
                         [233.5253448486328, 220.32708740234375], [234.2936248779297, 220.13198852539062],
                         [235.06190490722656, 219.9368896484375], [235.83018493652344, 219.74179077148438],
                         [236.5984649658203, 219.54669189453125], [237.3667449951172, 219.35159301757812],
                         [238.13502502441406, 219.15650939941406], [238.90330505371094, 218.96141052246094],
                         [239.67156982421875, 218.7663116455078], [240.43984985351562, 218.5712127685547],
                         [241.2081298828125, 218.37611389160156], [241.97640991210938, 218.18101501464844],
                         [242.74472045898438, 217.98590087890625], [243.51300048828125, 217.79080200195312],
                         [244.28128051757812, 217.595703125], [245.049560546875, 217.40060424804688],
                         [245.81784057617188, 217.20550537109375], [246.58612060546875, 217.01040649414062],
                         [247.35440063476562, 216.81532287597656], [248.1226806640625, 216.62022399902344],
                         [248.8909454345703, 216.4251251220703], [249.6592254638672, 216.2300262451172],
                         [250.42750549316406, 216.03492736816406], [251.19578552246094, 215.83982849121094],
                         [251.9640655517578, 215.6447296142578], [252.7323455810547, 215.4496307373047],
                         [253.50062561035156, 215.25453186035156], [254.26890563964844, 215.05943298339844],
                         [255.0371856689453, 214.8643341064453], [255.8054656982422, 214.6692352294922],
                         [256.57373046875, 214.47413635253906], [257.3420104980469, 214.27903747558594],
                         [258.11029052734375, 214.0839385986328], [258.8785705566406, 213.8888702392578],
                         [259.6468811035156, 213.69383239746094], [260.0309143066406, 213.27691650390625],
                         [259.81573486328125, 212.51402282714844], [259.6005554199219, 211.7510986328125],
                         [259.3853759765625, 210.98818969726562], [259.17022705078125, 210.2252960205078],
                         [258.9550476074219, 209.46238708496094], [258.7398681640625, 208.69949340820312],
                         [258.5246887207031, 207.93658447265625], [258.30950927734375, 207.17369079589844],
                         [258.0943603515625, 206.41079711914062], [257.8791809082031, 205.64788818359375],
                         [257.66400146484375, 204.88499450683594], [257.4488525390625, 204.12208557128906],
                         [257.2336730957031, 203.35919189453125], [257.01849365234375, 202.59628295898438],
                         [256.8033142089844, 201.83338928222656], [256.5881652832031, 201.07049560546875],
                         [256.37298583984375, 200.30758666992188], [256.1578063964844, 199.54469299316406],
                         [255.94264221191406, 198.7817840576172], [255.7274627685547, 198.01889038085938],
                         [255.51229858398438, 197.2559814453125], [255.297119140625, 196.4930877685547],
                         [255.0819549560547, 195.73019409179688], [254.8667755126953, 194.96728515625],
                         [254.651611328125, 194.2043914794922], [254.43643188476562, 193.4414825439453],
                         [254.2212677001953, 192.6785888671875], [254.00608825683594, 191.91567993164062],
                         [253.79092407226562, 191.1527862548828], [253.57574462890625, 190.389892578125],
                         [253.36058044433594, 189.62698364257812], [253.14540100097656, 188.8640899658203],
                         [252.93023681640625, 188.10118103027344], [252.71505737304688, 187.33828735351562],
                         [252.49989318847656, 186.57537841796875], [252.2847137451172, 185.81248474121094],
                         [252.06954956054688, 185.04959106445312], [251.8543701171875, 184.28668212890625],
                         [251.6392059326172, 183.52378845214844], [251.4240264892578, 182.76087951660156],
                         [251.2088623046875, 181.99798583984375], [250.99368286132812, 181.23507690429688],
                         [250.7785186767578, 180.47218322753906], [250.56333923339844, 179.70928955078125],
                         [250.34817504882812, 178.94638061523438], [250.13299560546875, 178.18348693847656],
                         [249.91783142089844, 177.4205780029297], [249.70265197753906, 176.65768432617188],
                         [249.48748779296875, 175.894775390625], [249.27230834960938, 175.1318817138672],
                         [249.05714416503906, 174.36898803710938], [248.8419647216797, 173.6060791015625],
                         [248.62680053710938, 172.8431854248047], [248.41162109375, 172.0802764892578],
                         [248.1964569091797, 171.3173828125], [247.9812774658203, 170.55447387695312],
                         [247.76611328125, 169.7915802001953], [247.55093383789062, 169.0286865234375],
                         [247.3357696533203, 168.26577758789062], [247.12059020996094, 167.5028839111328],
                         [246.90542602539062, 166.73997497558594], [246.69024658203125, 165.97708129882812],
                         [246.47508239746094, 165.2141876220703], [246.25990295410156, 164.45127868652344],
                         [246.04473876953125, 163.68838500976562], [245.82955932617188, 162.92547607421875],
                         [245.61439514160156, 162.16258239746094], [245.3992156982422, 161.39967346191406],
                         [245.18405151367188, 160.63677978515625], [244.9688720703125, 159.87388610839844],
                         [244.7537078857422, 159.11097717285156], [244.5385284423828, 158.34808349609375],
                         [244.3233642578125, 157.58517456054688], [244.10818481445312, 156.82228088378906],
                         [243.8930206298828, 156.0593719482422], [243.67784118652344, 155.29647827148438],
                         [243.46267700195312, 154.53358459472656], [243.24749755859375, 153.7706756591797],
                         [243.03233337402344, 153.00778198242188], [242.81715393066406, 152.244873046875],
                         [242.6020050048828, 151.48204040527344], [242.3868408203125, 150.71913146972656],
                         [242.19635009765625, 149.9562225341797], [242.41152954101562, 149.19332885742188],
                         [242.62669372558594, 148.430419921875], [242.8418731689453, 147.6675262451172],
                         [243.05703735351562, 146.90463256835938], [243.272216796875, 146.1417236328125],
                         [243.4873809814453, 145.3788299560547], [243.7025604248047, 144.6159210205078],
                         [243.917724609375, 143.85302734375], [244.13290405273438, 143.09011840820312],
                         [244.34808349609375, 142.3272247314453], [244.56324768066406, 141.5643310546875],
                         [244.77842712402344, 140.80142211914062], [244.99359130859375, 140.0385284423828],
                         [245.20877075195312, 139.275634765625], [245.42393493652344, 138.51272583007812],
                         [245.6391143798828, 137.74981689453125], [245.85427856445312, 136.98692321777344],
                         [246.0694580078125, 136.22402954101562], [246.2846221923828, 135.46112060546875],
                         [246.4998016357422, 134.69822692871094], [246.7149658203125, 133.93533325195312],
                         [246.93014526367188, 133.17242431640625], [247.14532470703125, 132.40953063964844],
                         [247.36048889160156, 131.64662170410156], [247.57566833496094, 130.88372802734375],
                         [247.79083251953125, 130.12081909179688], [248.00601196289062, 129.35792541503906],
                         [248.22117614746094, 128.59503173828125], [248.4363555908203, 127.83212280273438],
                         [248.65151977539062, 127.06922912597656], [248.86669921875, 126.30632781982422],
                         [249.0818634033203, 125.54342651367188], [249.2970428466797, 124.78052520751953],
                         [249.51220703125, 124.01762390136719], [249.72738647460938, 123.25473022460938],
                         [249.94256591796875, 122.4918212890625], [250.15773010253906, 121.72892761230469],
                         [250.37290954589844, 120.96602630615234], [250.58807373046875, 120.203125],
                         [250.80325317382812, 119.44022369384766], [251.01841735839844, 118.67732238769531],
                         [251.2335968017578, 117.9144287109375], [251.44876098632812, 117.15152740478516],
                         [251.6639404296875, 116.38862609863281], [251.8791046142578, 115.62572479248047],
                         [252.0942840576172, 114.86282348632812], [252.3094482421875, 114.09992980957031],
                         [252.52462768554688, 113.33702850341797], [252.73980712890625, 112.57412719726562],
                         [252.95497131347656, 111.81122589111328], [253.17015075683594, 111.04832458496094],
                         [253.38531494140625, 110.28543090820312], [253.60049438476562, 109.52252960205078],
                         [253.81565856933594, 108.75962829589844], [254.0308380126953, 107.9967269897461],
                         [254.24600219726562, 107.23382568359375], [254.461181640625, 106.4709243774414],
                         [254.6763458251953, 105.70802307128906], [254.8915252685547, 104.94512939453125],
                         [255.106689453125, 104.1822280883789], [255.32186889648438, 103.41932678222656],
                         [255.53704833984375, 102.65642547607422], [255.75221252441406, 101.89352416992188],
                         [255.96739196777344, 101.13063049316406], [256.18255615234375, 100.36772918701172],
                         [256.3977355957031, 99.60482788085938], [256.6129150390625, 98.84192657470703],
                         [256.82806396484375, 98.07902526855469], [257.0432434082031, 97.31612396240234],
                         [257.2584228515625, 96.55323028564453], [257.4736022949219, 95.79032897949219],
                         [257.6887512207031, 95.02742767333984], [257.9039306640625, 94.2645263671875],
                         [258.1191101074219, 93.50162506103516], [258.33428955078125, 92.73872375488281],
                         [258.5494689941406, 91.975830078125], [258.7646179199219, 91.21292877197266],
                         [258.97979736328125, 90.45002746582031], [259.1949768066406, 89.68712615966797],
                         [259.41015625, 88.92422485351562], [259.62530517578125, 88.16133117675781],
                         [259.8404846191406, 87.39839935302734], [260.0556640625, 86.63550567626953],
                         [259.5584716796875, 86.28388977050781], [258.7901916503906, 86.08880615234375],
                         [258.02191162109375, 85.89369201660156], [257.2536315917969, 85.69859313964844],
                         [256.4853515625, 85.50349426269531], [255.71707153320312, 85.30838775634766],
                         [254.94879150390625, 85.11328125], [254.18051147460938, 84.91818237304688],
                         [253.4122314453125, 84.72307586669922], [252.64395141601562, 84.5279769897461],
                         [251.87567138671875, 84.33287048339844], [251.10740661621094, 84.13777160644531],
                         [250.33912658691406, 83.94266510009766], [249.5708465576172, 83.74756622314453],
                         [248.8025665283203, 83.55245971679688], [248.03428649902344, 83.35736083984375],
                         [247.26602172851562, 83.1622543334961], [246.49774169921875, 82.96715545654297],
                         [245.72946166992188, 82.77204895019531], [244.961181640625, 82.57695007324219],
                         [244.19290161132812, 82.38184356689453], [243.42462158203125, 82.1867446899414],
                         [242.65634155273438, 81.99163818359375], [241.8880615234375, 81.79653930664062],
                         [241.1197967529297, 81.60143280029297], [240.3515167236328, 81.40633392333984],
                         [239.58323669433594, 81.21122741699219], [238.81495666503906, 81.01612854003906],
                         [238.04669189453125, 80.8210220336914], [237.27841186523438, 80.62592315673828],
                         [236.5101318359375, 80.43081665039062], [235.74185180664062, 80.2357177734375],
                         [234.97357177734375, 80.04061126708984], [234.20529174804688, 79.84551239013672],
                         [233.43701171875, 79.65040588378906], [232.66873168945312, 79.45530700683594],
                         [231.9004669189453, 79.26020050048828], [231.13218688964844, 79.06510162353516],
                         [230.36390686035156, 78.8699951171875], [229.5956268310547, 78.67489624023438],
                         [228.8273468017578, 78.47978973388672], [228.05908203125, 78.28468322753906],
                         [227.29080200195312, 78.08958435058594], [226.52252197265625, 77.89448547363281],
                         [225.75424194335938, 77.69937896728516], [224.9859619140625, 77.5042724609375],
                         [224.21768188476562, 77.30917358398438], [223.44940185546875, 77.11407470703125],
                         [222.68113708496094, 76.9189682006836], [221.91285705566406, 76.72386169433594],
                         [221.1445770263672, 76.52876281738281], [220.3762969970703, 76.33365631103516],
                         [219.60801696777344, 76.13855743408203], [218.83973693847656, 75.94345092773438],
                         [218.07147216796875, 75.74835205078125], [217.30319213867188, 75.5532455444336],
                         [216.534912109375, 75.35814666748047], [215.76663208007812, 75.16304016113281],
                         [214.99835205078125, 74.96794128417969], [214.23007202148438, 74.77283477783203],
                         [213.4617919921875, 74.5777359008789], [212.6935272216797, 74.38262939453125],
                         [211.9252471923828, 74.18753051757812], [211.15696716308594, 73.99242401123047],
                         [210.38868713378906, 73.79732513427734], [209.6204071044922, 73.60221862792969],
                         [208.85214233398438, 73.40711975097656], [208.0838623046875, 73.2120132446289],
                         [207.31558227539062, 73.01691436767578], [206.54730224609375, 72.82180786132812],
                         [205.77902221679688, 72.626708984375], [205.0107421875, 72.43160247802734],
                         [204.24246215820312, 72.23650360107422], [203.4741973876953, 72.04139709472656],
                         [202.70591735839844, 71.84629821777344], [201.93763732910156, 71.65119171142578],
                         [201.1693572998047, 71.45609283447266], [200.4010772705078, 71.260986328125],
                         [199.63279724121094, 71.06588745117188], [198.86453247070312, 70.87078094482422],
                         [198.09625244140625, 70.6756820678711], [197.32797241210938, 70.48057556152344],
                         [196.5596923828125, 70.28547668457031], [195.87559509277344, 69.9445571899414],
                         [195.32249450683594, 69.37676239013672], [194.76937866210938, 68.80897521972656],
                         [194.21627807617188, 68.24118041992188], [193.66317749023438, 67.67338562011719],
                         [193.1100616455078, 67.1055908203125], [192.5569610595703, 66.53779602050781],
                         [192.00384521484375, 65.97000122070312], [191.45074462890625, 65.40220642089844],
                         [190.89764404296875, 64.83441162109375], [190.34454345703125, 64.2666244506836],
                         [189.7914276123047, 63.698829650878906], [189.2383270263672, 63.131038665771484],
                         [188.68521118164062, 62.5632438659668], [188.13211059570312, 61.99544906616211],
                         [187.57901000976562, 61.42765808105469], [187.02590942382812, 60.85986328125],
                         [186.47279357910156, 60.29207229614258], [185.91969299316406, 59.72427749633789],
                         [185.3665771484375, 59.1564826965332], [184.8134765625, 58.58869171142578],
                         [184.2603759765625, 58.020896911621094], [183.70726013183594, 57.45310592651367],
                         [183.15415954589844, 56.885311126708984], [182.60104370117188, 56.3175163269043],
                         [182.04794311523438, 55.749725341796875], [181.49484252929688, 55.18193054199219],
                         [180.9417266845703, 54.6141357421875], [180.3886260986328, 54.04634475708008],
                         [179.8355255126953, 53.478553771972656], [179.28240966796875, 52.91075897216797],
                         [178.72930908203125, 52.34296417236328], [178.17620849609375, 51.775169372558594],
                         [177.6230926513672, 51.20737838745117], [177.0699920654297, 50.63958740234375],
                         [176.51687622070312, 50.07179260253906], [175.96377563476562, 49.503997802734375],
                         [175.41067504882812, 48.93620300292969], [174.85757446289062, 48.368412017822266],
                         [174.30445861816406, 47.800621032714844], [173.75135803222656, 47.232826232910156],
                         [173.1982421875, 46.66503143310547], [172.6451416015625, 46.09723663330078],
                         [172.092041015625, 45.52944564819336], [171.53892517089844, 44.96165466308594],
                         [170.98582458496094, 44.39385986328125], [170.43272399902344, 43.82606506347656],
                         [169.87960815429688, 43.258270263671875], [169.32650756835938, 42.69047927856445],
                         [168.77340698242188, 42.12268829345703], [168.2202911376953, 41.554893493652344],
                         [167.6671905517578, 40.987098693847656], [167.11407470703125, 40.41930389404297],
                         [166.56097412109375, 39.85151290893555], [166.00787353515625, 39.28371810913086],
                         [165.4547576904297, 38.71592712402344], [164.9016571044922, 38.14813232421875],
                         [164.3485565185547, 37.58033752441406], [163.79544067382812, 37.01254653930664],
                         [163.24234008789062, 36.44475173950195], [162.68923950195312, 35.87696075439453],
                         [162.13612365722656, 35.309165954589844], [161.58302307128906, 34.741371154785156],
                         [161.0299072265625, 34.173580169677734], [160.476806640625, 33.60578536987305],
                         [159.9237060546875, 33.037994384765625], [159.37059020996094, 32.47019958496094],
                         [158.81748962402344, 31.902406692504883], [158.26438903808594, 31.334613800048828],
                         [157.71127319335938, 30.76681900024414], [157.15817260742188, 30.199026107788086],
                         [156.60507202148438, 29.63123321533203], [156.0519561767578, 29.063440322875977],
                         [155.4988555908203, 28.495647430419922], [154.9457550048828, 27.927852630615234],
                         [154.39263916015625, 27.36005973815918], [153.83953857421875, 26.792266845703125],
                         [153.2864227294922, 26.22447395324707], [152.7333221435547, 25.656681060791016],
                         [152.1802215576172, 25.088886260986328], [151.62710571289062, 24.521093368530273]]

    # Border Tests
    # test_for_get_extended_bitmap(screen, extended_bitmap, cell_size)
    # test_for_get_printing_path(screen, border_printing_path)
    # draw_svg_path(screen)

    # GCODE
    file_name = ''
    # file_name = re.search(r'\d+_small_', infill_img_path).group(0)
    file_name = re.search(r'\d+_', infill_img_path).group(0)
    perimeter_type = 0
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
    get_gcode_file(print_options, printing_path, svg_border_points)

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
