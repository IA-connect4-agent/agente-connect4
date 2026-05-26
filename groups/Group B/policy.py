import numpy as np
import math
from connect4.policy import Policy
from connect4.connect_state import ConnectState


# Clase que representa un nodo dentro del árbol del MCTS
# Cada nodo guarda un estado del tablero y estadísticas de simulaciones
class _Node:

    # board -> tablero actual
    # player -> jugador al que le toca desde este estado
    # parent -> nodo padre
    # col_played -> columna usada para llegar a este nodo
    def __init__(self, board: np.ndarray, player: int, parent=None, col_played=None):

        self.board = board
        self.player = player
        self.parent = parent
        self.col_played = col_played

        # Estadísticas usadas por MCTS
        self.wins = 0.0
        self.visits = 0

        # Hijos del nodo actual
        self.children: list["_Node"] = []

        # Columnas que todavía no se han expandido
        state = ConnectState(board=board, player=player)

        # Si el juego terminó ya no hay movimientos posibles
        self.untried_cols = state.get_free_cols() if not state.is_final() else []

    # Revisa si ya se probaron todos los movimientos posibles
    def is_fully_expanded(self) -> bool:
        return len(self.untried_cols) == 0

    # Revisa si el estado actual ya es final
    def is_terminal(self) -> bool:
        return ConnectState(board=self.board, player=self.player).is_final()

    # Fórmula UCB1
    # Se usa para decidir qué hijo explorar
    # Balancea exploración y explotación
    def ucb1(self, c: float = math.sqrt(2)) -> float:

        # Si nunca se visitó, se prioriza explorarlo
        if self.visits == 0:
            return float("inf")

        return self.wins / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )


# Política basada en Monte Carlo Tree Search
class MCTry(Policy):

    # Tamaño del tablero
    ROWS = 6
    COLS = 7

    # budget -> cantidad de simulaciones por movimiento
    # c -> constante de exploración para UCB1
    def __init__(self, budget: int = 300, c: float = math.sqrt(2)):
        self.budget = budget
        self.c = c

    # No necesita entrenamiento previo
    # Todo se calcula mientras se juega
    def mount(self, *args) -> None:
        pass

    

    # Funciones auxiliares

    # Determina qué jugador debe jugar según el tablero
    def _current_player(self, board: np.ndarray) -> int:

        reds = int(np.sum(board == -1))
        yellows = int(np.sum(board == 1))

        # Si ambos tienen la misma cantidad de fichas,
        # le toca al jugador -1
        return -1 if reds == yellows else 1

    # Simula poner una ficha en una columna
    # y devuelve el nuevo tablero
    def _drop(self, board: np.ndarray, col: int, player: int) -> np.ndarray:

        b = board.copy()

        # Busca el espacio libre más abajo
        for r in range(self.ROWS - 1, -1, -1):
            if b[r, col] == 0:
                b[r, col] = player
                break

        return b

    # Revisa si hay un ganador en el tablero
    # Retorna:
    # -1 -> gana rojo
    #  1 -> gana amarillo
    #  0 -> nadie ganó todavía
    def _check_winner(self, board: np.ndarray) -> int:

        for r in range(self.ROWS):
            for c in range(self.COLS):

                p = board[r, c]

                # Espacio vacío
                if p == 0:
                    continue

                # Horizontal
                if c + 3 < self.COLS and all(board[r, c + i] == p for i in range(4)):
                    return p

                # Vertical
                if r + 3 < self.ROWS and all(board[r + i, c] == p for i in range(4)):
                    return p

                # Diagonal principal
                if r + 3 < self.ROWS and c + 3 < self.COLS and \
                   all(board[r + i, c + i] == p for i in range(4)):
                    return p

                # Diagonal inversa
                if r + 3 < self.ROWS and c - 3 >= 0 and \
                   all(board[r + i, c - i] == p for i in range(4)):
                    return p

        return 0

    # Devuelve las columnas que todavía tienen espacio
    def _free_cols(self, board: np.ndarray) -> list:
        return [c for c in range(self.COLS) if board[0, c] == 0]



    # Fases del MCTS


    # Fase 1: Selection
    # Baja por el árbol eligiendo siempre el mejor UCB1
    def _select(self, node: _Node) -> _Node:

        while not node.is_terminal() and node.is_fully_expanded():
            node = max(node.children, key=lambda n: n.ucb1(self.c))

        return node

    # Fase 2: Expansion
    # Agrega un nuevo hijo usando una columna no explorada
    def _expand(self, node: _Node) -> _Node:

        col = node.untried_cols.pop(
            np.random.randint(len(node.untried_cols))
        )

        new_board = self._drop(node.board, col, node.player)

        child = _Node(
            new_board,
            -node.player,
            parent=node,
            col_played=col
        )

        node.children.append(child)

        return child

    # Fase 3: Simulation
    # Juega aleatoriamente hasta terminar la partida
    def _simulate(self, node: _Node) -> int:

        rng = np.random.default_rng()

        board = node.board.copy()
        p = node.player

        while True:

            free = self._free_cols(board)

            # Empate
            if not free:
                return 0

            # Movimiento aleatorio
            col = int(rng.choice(free))

            for r in range(self.ROWS - 1, -1, -1):
                if board[r, col] == 0:
                    board[r, col] = p
                    break

            # Revisar ganador
            winner = self._check_winner(board)

            if winner != 0:
                return winner

            # Cambiar turno
            p = -p

    # Fase 4: Backpropagation
    # Actualiza estadísticas desde el nodo hoja hasta la raíz
    def _backprop(self, node: _Node, winner: int, root_player: int) -> None:

        while node is not None:

            node.visits += 1

            # Victoria
            if winner == root_player:
                node.wins += 1.0

            # Empate
            elif winner == 0:
                node.wins += 0.5


            node = node.parent

    # Revisiones rápidas antes de usar MCTS completo

    # Busca una victoria inmediata o bloquear al rival
    def _immediate_win_or_block(self, board: np.ndarray, player: int) -> int | None:

        free = self._free_cols(board)

        # Revisar si puedo ganar ya
        for col in free:
            if self._check_winner(self._drop(board, col, player)) == player:
                return col

        # Revisar si el rival puede ganar
        for col in free:
            if self._check_winner(self._drop(board, col, -player)) == -player:
                return col

        return None

    
    # Función principal del agente

    # Decide qué movimiento hacer
    def act(self, s: np.ndarray) -> int:

        player = self._current_player(s)
        free_cols = self._free_cols(s)

        # Tablero lleno
        if not free_cols:
            return 0

        # Solo queda una jugada posible
        if len(free_cols) == 1:
            return free_cols[0]

        # Atajos rápidos antes de correr MCTS
        quick = self._immediate_win_or_block(s, player)

        if quick is not None:
            return quick

        # Nodo raíz con el estado actual
        root = _Node(board=s, player=player)

        # Ejecutar simulaciones
        for _ in range(self.budget):

            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if not node.is_terminal():
                node = self._expand(node)

            # 3. Simulation
            winner = self._simulate(node)

            # 4. Backpropagation
            self._backprop(node, winner, player)

        # Elegir el hijo más visitado
        # Normalmente es el movimiento más estable
        best = max(root.children, key=lambda n: n.visits)

        return best.col_played
