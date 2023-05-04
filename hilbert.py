import math
import sys
import os
from typing import Tuple
import pygame
import cv2


# Utils
def line(ctx, from_: Tuple[int, int], to: Tuple[int, int]) -> None:
    pygame.draw.aaline(ctx, (255, 0, 0), from_, to)


# Hilbert Curve
def hindex2xy(hindex, curve_order):
    positions = [
        # 0:
        [0, 0],
        # 1:
        [0, 1],
        # 2:
        [1, 1],
        # 3:
        [1, 0]
    ]

    tmp = positions[last2bits(hindex)]
    hindex = (hindex >> 2)

    x = tmp[0]
    y = tmp[1]

    n = 4
    while n <= curve_order:
        n2 = n // 2

        case = last2bits(hindex)
        if case == 0:  # left-bottom
            tmp = x
            x = y
            y = tmp
        elif case == 1:  # left-upper
            x = x
            y = y + n2
        elif case == 2:  # right-upper
            x = x + n2
            y = y + n2
        elif case == 3:  # right-bottom
            tmp = y
            y = (n2 - 1) - x
            x = (n2 - 1) - tmp
            x = x + n2

        hindex = (hindex >> 2)
        n *= 2

    return [x, y]


def last2bits(x):
    return x & 3


# Squares Grid
def get_squares_grid(width: int, gap: int, offset=(0, 0)) -> \
        list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]]:
    # return: list of squares. Each square presented in the form of coordinates of four vertex
    if gap >= width:
        raise ValueError("Gap can not be >= width")
    if width % gap != 0:
        raise ValueError("Width must be divided without remainder by gap")

    grid_lines_count = width // gap
    squares_list = []
    for j in range(grid_lines_count):
        for i in range(grid_lines_count):
            # the coordinates of the four vertices of the square ABCD
            squares_list.append((
                (i * gap + offset[0], j * gap + offset[1]),
                ((i + 1) * gap + offset[0], j * gap + offset[1]),
                ((i + 1) * gap + offset[0], (j + 1) * gap + offset[1]),
                (i * gap + offset[0], (j + 1) * gap + offset[1])
            ))

    return squares_list


def draw_squares_grid(ctx, squares_grid, width: int, gap: int):
    for i, square in enumerate(squares_grid):
        # j is the number of the current vertex of the current square
        for j in range(len(square) - 1):
            # do not draw the bottom side of the square
            if j != 2:
                line(ctx, square[j], square[j + 1])
        # for the first column of the grid
        if i % (width // gap) == 0:
            # draw the left side of the square
            line(ctx, square[3], square[0])
        # for the last row of the grid
        if i >= (((width // gap) - 1) * (width // gap)):
            # draw the bottom side of the square
            line(ctx, square[2], square[3])


# Custom Hilbert Curve
def get_hilbert_dots(size, curve_order, coordinate_offset=(0, 0)):
    hilbert_dots = []

    block_size = size // curve_order
    offset = block_size // 2

    for i in range(curve_order * curve_order):
        hindex2 = hindex2xy(i, curve_order)
        hindex2_x = hindex2[0]
        hindex2_y = hindex2[1]
        hilbert_dots.append((
            hindex2_x * block_size + offset + coordinate_offset[0],
            hindex2_y * block_size + offset + coordinate_offset[1]
        ))

    return hilbert_dots


def draw_hilbert_curve(ctx, dots, curve_order):
    prev = dots[0]

    for i in range(curve_order * curve_order):
        curr = dots[i]
        line(ctx, prev, curr)
        prev = curr


# Fill Path
def get_fill_path():
    result = []
    return result


def draw_fill_path(ctx, squares_grid, width, gap, img):
    curve_order = 2

    hilbert_curves = []
    thresholds = []

    for r in range(0, img.shape[0], gap):
        for c in range(0, img.shape[1], gap):
            value = img[r:r + gap, c:c + gap].mean(axis=(0, 1))
            thresholds.append(value)
            # print(thresholds)
            # if value[0] < 230:
            #     print(f'{r}_{c}: {value[0]}')

    for i in range(len(squares_grid)):
        # random_threshold = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
        random_threshold = thresholds[i]
        # print(random_threshold)
        # if random_threshold >= 140:
        #     hilbert_curves.append(None)
        # else:
        #     dots = get_hilbert_dots(gap, curve_order * 1, squares_grid[i][0])
        #     draw_hilbert_curve(ctx, dots, curve_order * 1)
        #     hilbert_curves.append(dots)
        if random_threshold <= 25:
            dots = get_hilbert_dots(gap, curve_order * 4, squares_grid[i][0])
            draw_hilbert_curve(ctx, dots, curve_order * 4)
            hilbert_curves.append(dots)
        elif random_threshold <= 50:
            dots = get_hilbert_dots(gap, curve_order * 2, squares_grid[i][0])
            draw_hilbert_curve(ctx, dots, curve_order * 2)
            hilbert_curves.append(dots)
        elif random_threshold <= 150:
            dots = get_hilbert_dots(gap, curve_order, squares_grid[i][0])
            draw_hilbert_curve(ctx, dots, curve_order)
            hilbert_curves.append(dots)
        else:
            continue

    # print(hilbert_curves)
    # connect hilbert curves with each other
    for i in range(len(hilbert_curves) - 1):
        if hilbert_curves[i] is not None and hilbert_curves[i + 1] is not None:
            current_curve = hilbert_curves[i]
            # print(current_curve, '\n')
            next_curve = hilbert_curves[i + 1]
            start = current_curve[len(current_curve) - 1]
            stop = next_curve[0]
            dist = math.hypot(stop[0] - start[0], stop[1] - start[1])
            # old: ignore the last curve in each row of grid
            # if i % (width / gap) != ((width / gap) - 1) and dist < gap:
            if dist < gap:
                # line from the last dot of current curve to the first dot of next curve
                line(ctx, start, stop)
            # else:
            #     stop = current_curve[0]
            #     min_dist = math.hypot(stop[0] - start[0], stop[1] - start[1])
            #     min_dist_index = 0
            #     for j in range(len(current_curve) - 1):
            #         stop = current_curve[j]
            #         temp_dist = math.hypot(stop[0] - start[0], stop[1] - start[1])
            #         # print(stop)
            #         if temp_dist < min_dist:
            #             min_dist = temp_dist
            #             min_dist_index = j
            #
            #     stop = current_curve[min_dist_index]
            #     print(f'current curve: {current_curve}')
            #     print(f'start: {start}, stop: {stop}')
            #     print('\n\n')
            #     line(ctx, start, stop)
        elif hilbert_curves[i] is not None and hilbert_curves[i + 1] is None:
            current_curve = hilbert_curves[i]
            start = current_curve[len(current_curve) - 1]
            current_square = squares_grid[i]
            print(f'current square: {current_square}')
            B = current_square[1]
            C = current_square[2]
            x = B[0]
            y = start[1]
            stop = (x, y)
            # print(B)
            print(f'start: {start}, stop: {stop}')
            line(ctx, start, stop)

            is_last_curve_in_current_row = True
            l: int = i + 1
            index_coefficient = i // (width // gap) + 1
            print(f'i: {i}')
            print(f'index_coefficient: {index_coefficient}')
            stop_index = index_coefficient * (width // gap)
            print(f'stop_index: {stop_index}')

            for k in range(l, stop_index):
                print(f'k: {k}')
                if hilbert_curves[k] is not None:
                    is_last_curve_in_current_row = False
                    break

            if is_last_curve_in_current_row:
                line(ctx, stop, C)

            j = i + 1
            next_curve = hilbert_curves[j]
            while (next_curve is None) and (j < len(hilbert_curves)):
                next_curve = hilbert_curves[j]
                j += 1

            if next_curve is not None:
                stop = next_curve[0]
                dist = math.hypot(stop[0] - C[0], stop[1] - C[1])
                if is_last_curve_in_current_row and (dist < 20000 * gap):
                    line(ctx, C, stop)


# Main
def main():
    pygame.init()

    canvas = pygame.display.set_mode((1000, 1000))
    # canvas = pygame.display.set_mode((880, 880))
    pygame.display.set_caption("Squares Grid")

    ctx = canvas
    ctx.set_alpha(None)
    ctx.set_colorkey(None)
    ctx.fill((255, 255, 255))

    img_path = 'images/10.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('9.jpg', cv2.IMREAD_UNCHANGED)
    width, height = img.shape[0], img.shape[1]
    gap = 10

    squares_grid = get_squares_grid(width, gap, (20, 20))
    # draw_squares_grid(ctx, squares_grid, width, gap)
    draw_fill_path(ctx, squares_grid, width, gap, img)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONUP:
                x, y = pygame.mouse.get_pos()
                print(x, y)

        pygame.display.flip()


main()
