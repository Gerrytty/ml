import numpy as np
import matplotlib.pyplot as plt
import sys
import networkx as nx
import random

class Node:
    def __init__(self, from_index, to_index, weight):
        self.from_index = from_index
        self.to_index = to_index
        self.weight = weight
        self.coef = 0

    def __str__(self):
        return f"{self.from_index} {self.to_index} {self.weight}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.from_index == other.from_index and self.to_index == other.to_index

    def __hash__(self):
        return hash(str(self))

def get_min_node(nodes, edges, v_dict):
    min_weight = sys.maxsize
    min_node = None

    for node in nodes:
        # Пропускаем уже отмеченные ребра (против циклов)
        # if edges[node.from_index][node.to_index] == 0 or v_dict[node.from_index] == 0):
        if node.coef == 0 and (v_dict[node.to_index] == 0 or v_dict[node.from_index] == 0):
            # Ищем среди не изолированных вершин
            if node.weight < min_weight and (v_dict[node.from_index] != 0 or v_dict[node.to_index] != 0):
                min_weight = node.weight
                min_node = node

    return min_node


def knp(N, nodes):
    edges = np.zeros((N, N))

    # Создаем словарь индекс вершины:соединенные вершины
    nodes_dict = dict()
    # Словарь для степеней вершин
    v_dict = dict()

    for i in range(N):
        nodes_from = []
        for node in nodes:
            if node.from_index == i:
                nodes_from.append(node)
        nodes_dict[i] = nodes_from
        v_dict[i] = 0

    print(nodes_dict)

    # Выбираем ребро с наименьшим весом
    min_node = min(nodes, key=lambda x: x.weight)

    # Результирующий массив ребер
    result = []

    while True:

        if min_node is None:
            break

        print(f"Algorithm choose {min_node}")

        # Проверяем остались ли изолированные вершины
        ok = True
        for i in range(N):
            if v_dict[i] == 0:
                ok = False

        if ok:
            break

        result.append(min_node)

        # # Путь из вершины 1 в вершину 2 = путь из вершины 2 в вершину 1
        # edges[min_node.from_index][min_node.to_index] = 1
        # edges[min_node.to_index][min_node.from_index] = 1

        # Увеличиваем степени вершин, для этого ребра
        v_dict[min_node.from_index] += 1
        v_dict[min_node.to_index] += 1

        # Отмечаем ребро как уже пройденное
        min_node.coef = 1

        # Ищем найти изолированную точку, ближайшую к некоторой
        # неизолированной
        min_node = get_min_node(nodes, edges, v_dict)

    print(result)

    return result


def plot_graph(N, nodes, title: str, weighted=True):
    G = nx.Graph()

    for i in range(N):
        G.add_node(i)

    if weighted:
        G.add_weighted_edges_from((node.from_index, node.to_index, node.weight) for node in nodes)

    else:
        for node in nodes:
            G.add_edge(node.from_index, node.to_index)

    nx.draw_kamada_kawai(G, with_labels=True)
    plt.title(title)
    plt.show()


"""
Возвращает рандомный граф 
c N вершинами и вероятностью соединения между вершинами = prob_connection
"""
def init_random_graph(N, prob_connection=0.7):

    graph_edges = []

    graph = nx.erdos_renyi_graph(N, prob_connection)

    # Если наш рандомный граф получился хоть с одной изолированной вершиной
    if any([x[1] for x in graph.degree()]) == 0:
        return init_random_graph(N, prob_connection)

    edges_list = list(graph.edges)

    for edge in edges_list:
        random_weight = np.random.randint(1, 10)
        graph_edges.append(Node(edge[0], edge[1], random_weight))
        # graph_edges.append(Node(edge[1], edge[0], random_weight))

    return list(set(graph_edges))


def get_k_clusters(K, N, nodes, plot=True, user_input=False):
    result = knp(N, nodes)

    if plot:
        # Рисуем финальный граф (остовное дерево)
        plot_graph(N, result, "Финальный граф")

    # Сортируем в порядке убывания
    result.sort(key=lambda x: x.weight, reverse=True)

    # Графический способ определения числа кластеров
    if user_input:
        start = 0
        for r in result:
            plt.plot([start, start + r.weight], [0, 0], linewidth=12)
            start += r.weight
        plt.show()

        user_k = input("Ведите k (Чтобы пропустить этот шаг и взять дефолтное число k нажмите Enter): ")

        if user_k.isnumeric():
            K = int(user_k)

    print(f"k = {K}")

    # Список смежности для финального графа
    # Убираем k - 1 самых длинных ребер
    result = result[K - 1:]

    print(result)

    if plot:
        plot_graph(N, result, "К - cluster граф")

    return result


"""
requirement:
библиотека networkx
"""
if __name__ == "__main__":
    # Кол-во вершин в графе
    N = 10

    # Кол-во кластеров
    k = 2

    nodes = init_random_graph(N, 0.5)

    # Рисуем исходный граф
    plot_graph(N, nodes, "Исходный граф", False)

    clusters = get_k_clusters(k, N, nodes)