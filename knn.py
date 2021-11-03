from dataclasses import dataclass
import numpy as np
import pygame
import random
from scipy.stats import mode
import itertools


@dataclass
class Point:
    x: int
    y: int
    cluster: int
    color: str = 'red'


def dist(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def set_points(number_of_class_el, number_of_classes, colors, n_samples=10):
    data = []
    # colors = ['red', 'green', 'blue']
    for classNum in range(number_of_classes):
        center_x, center_y = random.randint(50, 550), random.randint(50, 350)
        for rowNum in range(number_of_class_el):
            data.append(Point(random.gauss(center_x, 20), random.gauss(center_y, 20), classNum, colors[classNum]))
            # print(cm.rainbow(np.linspace(0, 1, numberOfClassEl)))

    return data


"""
Функция для обучения и подбора оптимального количества соседей
"""
def train(new_point, true_point_class, colors):

    # Делаем точку, принадлежащей этому ее реальному классу
    new_point.cluster = true_point_class
    new_point.color = colors[true_point_class]

    # Проверяем максимум maximum_of_neighbors соседей и минимум minimum_of_neighbors
    for k in range(minimum_of_neighbors, maximum_of_neighbors):

        k_nearest_neighbors = find_k_nearest_neighbors(points, new_point, k)
        classes = [point.cluster for point in k_nearest_neighbors]
        predicted_point_class = mode(classes)[0][0]

        print(f"I guess point belong to {predicted_point_class} class with k = {k}")

        # Если с данным k класс предсказан верно, инкрементируем счетчик данного k
        if predicted_point_class == true_point_class:
            k_neighbors[k - 1] += 1

    # Добавляем точку с правильным кластером в общий массив точек
    points.append(new_point)


# Ищем k ближайших соседей
def find_k_nearest_neighbors(points, point, k):
    # Сортируем все точки по расстоянию до данной точки
    # И возвращаем первые k точек
    return sorted(points, key=lambda p: dist(p, point))[:k]


"""
Функция для предсказания номера кластера
"""
def predict_class(points, point, k, colors):

    # Ищем k ближайших соседей
    k_nearest_neighbors = find_k_nearest_neighbors(points, point, k)

    # Переводим в массив предсказанных классов
    classes = [point.cluster for point in k_nearest_neighbors]

    # Берем наиболее часто встречающийся класс
    predicted_point_class = mode(classes)[0][0]

    # Делаем точку, принадлежащей этому классу
    point.cluster = predicted_point_class
    point.color = colors[predicted_point_class]

    # Добавляем в общий массив точек
    points.append(point)


def draw_pygame(colors):
    pygame.init()

    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    play = True

    myfont = pygame.font.SysFont('Times New Roman', 25)
    textsurface = myfont.render('Введите номер кластера, а после нажмите enter',
                                False, (0, 0, 0))

    wait_cluster_num = False
    true_cluster = ""
    p = None

    while play:

        screen.fill('WHITE')

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                play = False

            if event.type == pygame.MOUSEBUTTONDOWN:

                """
                Train events
                Левая клавиша мыши
                Не даем человеку добавлять точки до тех пор,
                Пока не будет введен правильный номер кластера для точки
                """
                if event.button == 1 and not wait_cluster_num:
                    pygame.draw.circle(screen, 'yellow', event.pos, 5)
                    p = Point(event.pos[0], event.pos[1], -1, "yellow")

                    # Надпись, что нужно ввести номер кластера
                    screen.blit(textsurface, (0, 0))

                    wait_cluster_num = True
                    true_cluster = ""

                # Правая клавиша мыши
                if event.button == 3 and not wait_cluster_num:
                    pygame.draw.circle(screen, 'pink', event.pos, 5)
                    # k соседей с максимальным количеством плюсов, полученных во время обучения
                    optimal_k = np.argmax(k_neighbors)

                    # Если обучения не было
                    if optimal_k < minimum_of_neighbors:
                        print(f"(!) Пропущен шаг обучения берем дефолтное минимальное число соседей {minimum_of_neighbors}")
                        optimal_k = minimum_of_neighbors

                    # Предсказываем класс
                    predict_class(points, Point(event.pos[0], event.pos[1], -1, "pink"), optimal_k, colors)

            if event.type == pygame.KEYDOWN and wait_cluster_num:

                # Если нажата клавиша enter
                if event.key == pygame.K_RETURN:

                    if true_cluster.isnumeric():

                        if int(true_cluster) >= N:
                            print(f"Введенный кластер {true_cluster} превосходит число заданных кластеров {N}")
                            true_cluster = ""

                        else:
                            # Обучаем оптимальное кол-во соседей
                            train(p, int(true_cluster), colors)
                            p = None
                            wait_cluster_num = False
                    else:
                        true_cluster = ""
                        print(f"Введенный кластер {true_cluster} не является числом")

                # Если нажата клавиша от 0 до 10 (unicode)
                if int(event.key) in range(48, 58):
                    # Формируем номер класса, введенный пользователем
                    true_cluster += f"{int(event.key) - 48}"

        # Отрисовываем все точки
        for point in points:
            pygame.draw.circle(screen, point.color, (point.x, point.y), 5)

        """
        Рисуем желтую точку, до тех пор пока пользователь
        не введет правильный номер кластера
        после этого на этой точке обучается оптимальное число соседей
        точка окрашивается в свой правильный цвет и добавляется в общий массив точек
        """
        if p is not None:
            screen.blit(textsurface, (0, 0))
            pygame.draw.circle(screen, "yellow", (p.x, p.y), 5)
        pygame.display.update()


"""
Функция возвращает n разных цветов, но они могут быть не очень разными
"""
def color_generate(n, print_color=True):
    l = [0, 255, 0]

    list_of_colors = []

    i = 2
    while len(list_of_colors) < n:
        list_of_colors = list(set(itertools.combinations_with_replacement(l, 3)))
        # Удаляем желтый и белый
        list_of_colors.remove((255, 255, 0))
        list_of_colors.remove((255, 255, 255))
        l.append(255 / i)
        i += 2

    if print_color:
        from webcolors import rgb_to_name
        # Пишем сгенерированные цвета кластеров
        for i, color in enumerate(list_of_colors[:N]):
            try:
                print(f"cluster {i} - color {color} ({rgb_to_name((int(color[0]), int(color[1]), int(color[2])))})")
            except ValueError:
                print(f"cluster {i} - color {color} (unknown)")

    return list_of_colors[:N]


"""
Requirement: библиотека webcolors для перевода ргб в название цвета
Или просто поменять colors = color_generate(N) 
на colors = color_generate(N, False)

Для того чтобы обучить k соседей нужно:
Поставить точку, нажав левую клавишу,
затем ввести правильный номер кластера (верхние цифры на клавиатуре),
после того как ввели число, нужно нажать Enter
(Число может быть и двузначным)
((!) Нумерация кластеров идет с нуля)

Для того чтобы посмотреть предсказывание класса
по обученным k-оптимальным соседям нужно:
Поставить точку нажав на правую кнопку мыши
"""
if __name__ == '__main__':

    # Кол-во кластеров
    N = 3

    colors = ['red', 'green', 'blue']

    if N > 3:
        colors = color_generate(N)
    else:
        for i, color in enumerate(colors):
            print(f"cluster {i} - color {color}")

    # Нужно для обучения, сколько максимум соседних точек мы проверяем
    maximum_of_neighbors = 30
    # Сколько минимум соседних точек мы проверяем
    minimum_of_neighbors = 5

    # Счетчики правильных ответов для каждого k-количества соседей
    k_neighbors = np.zeros(maximum_of_neighbors)

    list_of_colors = [(255, 122, 0), (255, 0, 255), (0, 255, 255),
                      (0, 255, 0), (255, 0, 0), (0, 0, 255),
                      (0, 0, 122), (0, 122, 0), (122, 255, 0), (255, 122, 122)]

    pnt = 10
    points = set_points(pnt, N, colors)
    draw_pygame(colors)
