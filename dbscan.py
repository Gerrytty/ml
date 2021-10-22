import pygame
import random
import numpy as np

points = []

RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)


class Point:
    def __init__(self, x, y, color=BLACK, cluster=0):
        self.x = x
        self.y = y
        self.color = color
        self.cluster = cluster


def add_points(event_pos, r=4):

    point = Point(event_pos[0], event_pos[1])
    points.append(point)

    k = random.randint(1, 4)

    for i in range(k):
        d = random.randint(2 * r, 5 * r)
        alpha = random.random() * 2 * np.pi
        x_new = point.x + d * np.sin(alpha)
        y_new = point.y + d * np.cos(alpha)
        points.append(Point(x_new, y_new))


def pygame_draw(r=4):
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill('WHITE')
    pygame.display.update()
    FPS = 60
    clock = pygame.time.Clock()
    play = True
    while play:
        screen.fill('WHITE')
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    add_points(event.pos)
            if event.type == pygame.KEYDOWN:
                # R button on keyboard
                if event.key == pygame.K_r:
                    set_color()
                # С button on keyboard
                if event.key == pygame.K_c:
                    set_color()
                    set_new_color()
        for point in points:
            pygame.draw.circle(screen, point.color, (point.x, point.y), r)
        pygame.display.update()
        clock.tick(FPS)


def set_color():

    # сбрасываем значения, чтобы можно было
    # добавлять новые точки (и новые кластеры) на экран
    # после уже кластеризованных точек

    for point in points:
        point.cluster = 0
        point.color = BLACK

    for point in points:
        neighb = 0
        has_neighbour = False
        for neighbor in points:
            if dist(point, neighbor) <= eps:
                neighb += 1
                # Если мы уже нашли min_pts соседей, то нет смысла искать еще
                if neighb >= min_pts + 1:
                    has_neighbour = True
                    break
        if has_neighbour:
            point.color = GREEN

    for i, point in enumerate(points):
        if point.color != GREEN:
            for j, neighbor in enumerate(points):
                if i != j and neighbor.color == GREEN and dist(point, neighbor) < eps:
                    point.color = YELLOW

    for point in points:
        if point.color != GREEN and point.color != YELLOW:
            point.color = RED


def set_cluster():

    cluster_num = 1

    green_points = []

    for point in points:
        if point.color == GREEN:
            green_points.append(point)

    # Ищем всех соседей зеленых точек
    for green_point in green_points:
        if green_point.cluster == 0:
            green_point.cluster = cluster_num
            cluster_num += 1
            # рекурсивно
            get_neighbours(green_points, green_point)

    yellow_points = []

    for point in points:
        if point.color == YELLOW:
            yellow_points.append(point)

    # Ищем ближайший кластер для желтой точки
    for yellow_point in yellow_points:
        min_dist = dist(yellow_point, green_points[0])
        yellow_point.cluster = green_points[0].cluster
        for green_point in green_points:
            if dist(yellow_point, green_point) < min_dist:
                yellow_point.cluster = green_point.cluster
                min_dist = dist(yellow_point, green_point)

    return cluster_num


def get_neighbours(points, point):
    for neighbor in points:
        if neighbor.cluster == 0:
            if dist(point, neighbor) < eps:
                neighbor.cluster = point.cluster
                get_neighbours(points, neighbor)


def set_new_color():
    k_clusters = set_cluster()
    print(f"clusters num = {k_clusters - 1}")
    ret = colors(k_clusters + 1)

    for point in points:
        if point.cluster != 0:
            point.color = ret[point.cluster]


def dist(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


# https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
def colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r, g, b))

    return ret


"""
Чтобы запустить функцию и видеть как точки окрашиваются в зеленый и желтый
нужно нажать клавишу R
Чтобы посмотреть на новые цвета кластеров нужно нажать C
Клавиша С повторяет функционал клавиши R, поэтому ее нажимать не обязательно
PS язык раскладки клавиатуры должен быть английский
"""
if __name__ == "__main__":
    r = 4
    min_pts = 3
    eps = 15
    pygame_draw(r)