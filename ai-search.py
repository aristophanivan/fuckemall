import heapq
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class CellType(Enum):
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3
    PATH = 4
    VISITED = 5


@dataclass(order=True)
class Node:
    """Узел для приоритетной очереди"""
    priority: float
    position: Tuple[int, int] = field(compare=False)
    g_cost: float = field(compare=False)  # Стоимость от старта
    h_cost: float = field(compare=False)  # Эвристика до цели
    parent: Optional['Node'] = field(compare=False, default=None)


class PathFinder:
    """Класс для поиска пути с использованием A* и Greedy алгоритмов"""
    
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        # 8 направлений: вверх, вниз, влево, вправо + диагонали
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Основные
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Диагональные
        ]
    
    def heuristic_manhattan(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Манхэттенское расстояние"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def heuristic_euclidean(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Евклидово расстояние"""
        return ((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2) ** 0.5
    
    def heuristic_chebyshev(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Расстояние Чебышева (для 8 направлений)"""
        return max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))
    
    def is_valid(self, row: int, col: int) -> bool:
        """Проверка валидности позиции"""
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row][col] != CellType.WALL.value)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Получить соседей с весами перехода"""
        neighbors = []
        for dr, dc in self.directions:
            new_row, new_col = pos[0] + dr, pos[1] + dc
            if self.is_valid(new_row, new_col):
                # Диагональное движение стоит √2 ≈ 1.414
                cost = 1.414 if dr != 0 and dc != 0 else 1.0
                neighbors.append(((new_row, new_col), cost))
        return neighbors
    
    def reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Восстановление пути от цели к старту"""
        path = []
        current = node
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Разворачиваем путь
    
    def a_star(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int],
        heuristic: Callable = None
    ) -> Tuple[List[Tuple[int, int]], Dict]:
        """
        Алгоритм A*
        
        f(n) = g(n) + h(n)
        - g(n): реальная стоимость от старта до n
        - h(n): эвристическая оценка от n до цели
        
        A* гарантирует оптимальный путь при допустимой эвристике
        """
        if heuristic is None:
            heuristic = self.heuristic_manhattan
        
        stats = {
            'nodes_expanded': 0,
            'nodes_generated': 0,
            'path_length': 0,
            'path_cost': 0.0
        }
        
        # Приоритетная очередь (min-heap)
        open_set = []
        start_node = Node(
            priority=heuristic(start, goal),
            position=start,
            g_cost=0,
            h_cost=heuristic(start, goal)
        )
        heapq.heappush(open_set, start_node)
        
        # Множество посещённых узлов
        closed_set = set()
        
        # Словарь лучших g-стоимостей
        g_scores = {start: 0}
        
        while open_set:
            current = heapq.heappop(open_set)
            stats['nodes_expanded'] += 1
            
            # Нашли цель
            if current.position == goal:
                path = self.reconstruct_path(current)
                stats['path_length'] = len(path)
                stats['path_cost'] = current.g_cost
                return path, stats
            
            if current.position in closed_set:
                continue
            
            closed_set.add(current.position)
            
            # Обрабатываем соседей
            for neighbor_pos, move_cost in self.get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue
                
                new_g = current.g_cost + move_cost
                
                # Если нашли лучший путь к соседу
                if neighbor_pos not in g_scores or new_g < g_scores[neighbor_pos]:
                    g_scores[neighbor_pos] = new_g
                    h = heuristic(neighbor_pos, goal)
                    f = new_g + h
                    
                    neighbor_node = Node(
                        priority=f,
                        position=neighbor_pos,
                        g_cost=new_g,
                        h_cost=h,
                        parent=current
                    )
                    heapq.heappush(open_set, neighbor_node)
                    stats['nodes_generated'] += 1
        
        return [], stats  # Путь не найден
    
    def greedy_best_first(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int],
        heuristic: Callable = None
    ) -> Tuple[List[Tuple[int, int]], Dict]:
        """
        Жадный алгоритм поиска (Greedy Best-First Search)
        
        f(n) = h(n)
        
        Использует ТОЛЬКО эвристику, игнорируя реальную стоимость.
        Быстрее A*, но НЕ гарантирует оптимальный путь!
        """
        if heuristic is None:
            heuristic = self.heuristic_manhattan
        
        stats = {
            'nodes_expanded': 0,
            'nodes_generated': 0,
            'path_length': 0,
            'path_cost': 0.0
        }
        
        open_set = []
        start_node = Node(
            priority=heuristic(start, goal),  # Только h(n)!
            position=start,
            g_cost=0,
            h_cost=heuristic(start, goal)
        )
        heapq.heappush(open_set, start_node)
        
        visited = set()
        
        while open_set:
            current = heapq.heappop(open_set)
            stats['nodes_expanded'] += 1
            
            if current.position == goal:
                path = self.reconstruct_path(current)
                stats['path_length'] = len(path)
                stats['path_cost'] = current.g_cost
                return path, stats
            
            if current.position in visited:
                continue
            
            visited.add(current.position)
            
            for neighbor_pos, move_cost in self.get_neighbors(current.position):
                if neighbor_pos not in visited:
                    h = heuristic(neighbor_pos, goal)
                    neighbor_node = Node(
                        priority=h,  # Приоритет = только эвристика
                        position=neighbor_pos,
                        g_cost=current.g_cost + move_cost,
                        h_cost=h,
                        parent=current
                    )
                    heapq.heappush(open_set, neighbor_node)
                    stats['nodes_generated'] += 1
        
        return [], stats


class GridVisualizer:
    """Визуализация сетки и пути в консоли"""
    
    SYMBOLS = {
        CellType.EMPTY: '·',
        CellType.WALL: '█',
        CellType.START: 'S',
        CellType.GOAL: 'G',
        CellType.PATH: '★',
        CellType.VISITED: '○'
    }
    
    @staticmethod
    def visualize(
        grid: List[List[int]], 
        path: List[Tuple[int, int]] = None,
        start: Tuple[int, int] = None,
        goal: Tuple[int, int] = None
    ) -> str:
        """Создать строковое представление сетки"""
        display = [row[:] for row in grid]  # Копия
        
        if path:
            for pos in path:
                if pos != start and pos != goal:
                    display[pos[0]][pos[1]] = CellType.PATH.value
        
        if start:
            display[start[0]][start[1]] = CellType.START.value
        if goal:
            display[goal[0]][goal[1]] = CellType.GOAL.value
        
        result = []
        for row in display:
            line = ' '.join(
                GridVisualizer.SYMBOLS.get(CellType(cell), '?') 
                for cell in row
            )
            result.append(line)
        
        return '\n'.join(result)


def create_maze(rows: int, cols: int) -> List[List[int]]:
    """Создать тестовый лабиринт"""
    grid = [[CellType.EMPTY.value for _ in range(cols)] for _ in range(rows)]
    
    # Добавляем стены
    walls = [
        # Вертикальная стена
        (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
        # Горизонтальная стена
        (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
        # Ещё стены
        (7, 5), (7, 6), (7, 7),
        (1, 7), (2, 7), (3, 7),
        (8, 2), (8, 3), (8, 4),
    ]
    
    for r, c in walls:
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = CellType.WALL.value
    
    return grid


def compare_algorithms():
    """Сравнение A* и Greedy алгоритмов"""
    
    print("=" * 60)
    print("    СРАВНЕНИЕ АЛГОРИТМОВ ПОИСКА: A* vs GREEDY")
    print("=" * 60)
    
    # Создаём лабиринт
    grid = create_maze(12, 15)
    start = (1, 1)
    goal = (10, 13)
    
    finder = PathFinder(grid)
    visualizer = GridVisualizer()
    
    print("\n📋 ИСХОДНЫЙ ЛАБИРИНТ:")
    print("-" * 40)
    print(visualizer.visualize(grid, start=start, goal=goal))
    print("\nЛегенда: S=старт, G=цель, █=стена, ·=пустая клетка")
    
    # A* поиск
    print("\n" + "=" * 60)
    print("🔍 АЛГОРИТМ A*")
    print("=" * 60)
    
    start_time = time.perf_counter()
    path_astar, stats_astar = finder.a_star(start, goal)
    time_astar = (time.perf_counter() - start_time) * 1000
    
    if path_astar:
        print(visualizer.visualize(grid, path_astar, start, goal))
        print(f"\n📊 Статистика A*:")
        print(f"   • Узлов раскрыто: {stats_astar['nodes_expanded']}")
        print(f"   • Узлов сгенерировано: {stats_astar['nodes_generated']}")
        print(f"   • Длина пути: {stats_astar['path_length']} шагов")
        print(f"   • Стоимость пути: {stats_astar['path_cost']:.2f}")
        print(f"   • Время: {time_astar:.3f} мс")
    else:
        print("❌ Путь не найден!")
    
    # Greedy поиск
    print("\n" + "=" * 60)
    print("🏃 ЖАДНЫЙ АЛГОРИТМ (GREEDY)")
    print("=" * 60)
    
    start_time = time.perf_counter()
    path_greedy, stats_greedy = finder.greedy_best_first(start, goal)
    time_greedy = (time.perf_counter() - start_time) * 1000
    
    if path_greedy:
        print(visualizer.visualize(grid, path_greedy, start, goal))
        print(f"\n📊 Статистика Greedy:")
        print(f"   • Узлов раскрыто: {stats_greedy['nodes_expanded']}")
        print(f"   • Узлов сгенерировано: {stats_greedy['nodes_generated']}")
        print(f"   • Длина пути: {stats_greedy['path_length']} шагов")
        print(f"   • Стоимость пути: {stats_greedy['path_cost']:.2f}")
        print(f"   • Время: {time_greedy:.3f} мс")
    else:
        print("❌ Путь не найден!")
    
    # Сравнение
    print("\n" + "=" * 60)
    print("📈 СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 60)
    
    if path_astar and path_greedy:
        print(f"""
┌─────────────────────┬────────────┬────────────┐
│ Метрика             │    A*      │   Greedy   │
├─────────────────────┼────────────┼────────────┤
│ Узлов раскрыто      │ {stats_astar['nodes_expanded']:^10} │ {stats_greedy['nodes_expanded']:^10} │
│ Длина пути          │ {stats_astar['path_length']:^10} │ {stats_greedy['path_length']:^10} │
│ Стоимость пути      │ {stats_astar['path_cost']:^10.2f} │ {stats_greedy['path_cost']:^10.2f} │
│ Время (мс)          │ {time_astar:^10.3f} │ {time_greedy:^10.3f} │
└─────────────────────┴────────────┴────────────┘

💡 ВЫВОДЫ:
   • A* гарантирует ОПТИМАЛЬНЫЙ путь
   • Greedy обычно быстрее, но путь может быть НЕОПТИМАЛЬНЫМ
   • A* раскрывает больше узлов для гарантии оптимальности
""")


def interactive_demo():
    """Интерактивная демонстрация с разными эвристиками"""
    
    print("\n" + "=" * 60)
    print("🎯 СРАВНЕНИЕ ЭВРИСТИК ДЛЯ A*")
    print("=" * 60)
    
    grid = create_maze(12, 15)
    start = (1, 1)
    goal = (10, 13)
    finder = PathFinder(grid)
    
    heuristics = [
        ("Манхэттен", finder.heuristic_manhattan),
        ("Евклид", finder.heuristic_euclidean),
        ("Чебышев", finder.heuristic_chebyshev),
    ]
    
    print("\nРезультаты для разных эвристик:")
    print("-" * 50)
    
    for name, h_func in heuristics:
        path, stats = finder.a_star(start, goal, heuristic=h_func)
        print(f"\n{name}:")
        print(f"  Раскрыто узлов: {stats['nodes_expanded']}")
        print(f"  Стоимость пути: {stats['path_cost']:.2f}")


# Пример использования для игрового движка
class GamePathfinder:
    """Практический пример для игры"""
    
    def __init__(self, world_width: int, world_height: int):
        self.grid = [[0] * world_width for _ in range(world_height)]
        self.finder = PathFinder(self.grid)
    
    def add_obstacle(self, x: int, y: int):
        """Добавить препятствие"""
        if 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0]):
            self.grid[y][x] = CellType.WALL.value
    
    def find_path(
        self, 
        start_x: int, start_y: int, 
        goal_x: int, goal_y: int,
        use_astar: bool = True
    ) -> List[Tuple[int, int]]:
        """Найти путь между точками"""
        self.finder.grid = self.grid
        start = (start_y, start_x)
        goal = (goal_y, goal_x)
        
        if use_astar:
            path, _ = self.finder.a_star(start, goal)
        else:
            path, _ = self.finder.greedy_best_first(start, goal)
        
        # Конвертируем обратно в (x, y)
        return [(col, row) for row, col in path]


if __name__ == "__main__":
    compare_algorithms()
    interactive_demo()
    
    print("\n" + "=" * 60)
    print("🎮 ПРИМЕР ДЛЯ ИГРЫ")
    print("=" * 60)
    
    game = GamePathfinder(10, 10)
    
    # Добавляем препятствия
    for x in range(2, 8):
        game.add_obstacle(x, 5)
    
    # Ищем путь
    path = game.find_path(1, 1, 8, 8)
    print(f"\nПуть для игрового персонажа: {path}")