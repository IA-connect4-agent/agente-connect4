import numpy as np
from connect4.policy import Policy
from typing import override
from connect4.connect_state import ConnectState


class RulesValues(Policy):
    consecutive = 4
    values = {
        'own_4': 1000,
        'own_3': 100,
        'own_2': 10,
        'own_1': 1,
        'rival_3': -200,
        'rival_2': -10
    }
    @override
    def mount(self) -> None:
        pass

    @override
    def act(self, s: np.ndarray) -> int:
        available_cols = [c for c in range(7) if s[0, c] == 0]

         # Detectar quién soy
        reds    = int(np.sum(s == -1))
        yellows = int(np.sum(s == 1))
        me    = -1 if reds == yellows else 1
        rival =  1 if me == -1 else -1

        # Reglas de Agente Rules 

        # 1. Ganar si puedo
        for c in available_cols:
            if self._wins_after(s, c, me):
                return c

        # 2. Bloquear victoria inmediata del rival
        for c in available_cols:
            if self._wins_after(s, c, rival):
                return c

        # 3. Evitar jugadas que permiten perder en el siguiente turno
        safe_moves = []

        for c in available_cols:
            temp = s.copy()
            r = self._free_row(temp, c)
            if r is None:
                continue

            temp[r, c] = me

            losing = False

            for c2 in range(7):
                r2 = self._free_row(temp, c2)
                if r2 is None:
                    continue

                temp2 = temp.copy()
                temp2[r2, c2] = rival

                if ConnectState(temp2).get_winner() == rival:
                    losing = True
                    break

            if not losing:
                safe_moves.append(c)

        available_cols = safe_moves if safe_moves else available_cols

        # 4. Evitar jugadas que crean forks del rival (extra seguridad)
        safe_moves = [
            c for c in available_cols
            if not self._creates_fork(s, c, me, rival)
        ]

        if safe_moves:
            available_cols = safe_moves

        # Mirar en que estado esta el tablero y elegir la mejor jugada

        best_col = available_cols[0]
        best_score = -float('inf')

        for c in available_cols:
            temp = s.copy()
            row = self._free_row(temp, c)
            if r is None: 
                continue 
            temp[row, c] = me
            score = self.evaluate_board(temp, me, rival)
            if score > best_score:
                best_score = score
                best_col = c

        return best_col


    def _wins_after(self, s: np.ndarray, col: int, player: int) -> bool:
        temp = s.copy()
        r = self._free_row(temp, col)
        if r is None:
            return False
        temp[r, col] = player
        return ConnectState(temp).get_winner() == player


    def _creates_fork(self, s: np.ndarray, col: int,
                      me: int, rival: int) -> bool:
        temp = s.copy()
        r = self._free_row(temp, col)
        if r is None:
            return True

        temp[r, col] = me

        winning_moves = 0
        for c in range(7):
            r2 = self._free_row(temp, c)
            if r2 is not None:
                temp2 = temp.copy()
                temp2[r2, c] = rival
                if ConnectState(temp2).get_winner() == rival:
                    winning_moves += 1

        return winning_moves >= 2


    def _free_row(self, board: np.ndarray, col: int) -> int | None:
        for row in reversed(range(board.shape[0])):
            if board[row, col] == 0:
                return row
        return None
    
    def  evaluate_board(self, board: np.ndarray, me: int, rival: int) -> int:
        score = 0 
        rows, cols = board.shape

        # Revisar Horizontal 
        for r in range(rows):
            for c in range(cols - 3):
                piece = board[r, c:c+4]
                score += self.evaluate_consecutive(piece, me, rival)

        # Revisar Vertical
        for c in range(cols):
            for r in range(rows - 3):
                piece = board[r:r+4, c]
                score += self.evaluate_consecutive(piece, me, rival)

        # Revisar Diagonal Derecha abajo 
        for r in range(rows - 3):
            for c in range(cols - 3):
                piece = [board[r+i, c+i] for i in range(4)]
                score += self.evaluate_consecutive(piece, me, rival)

        # Revisar Diagonal izquierda abajo 
        for r in range(rows-3):    
            for c in range(3, cols):
                piece = [board[r+i, c-i] for i in range(4)]
                score += self.evaluate_consecutive(piece, me, rival)    
        return score
    
    def evaluate_consecutive(self, piece: np.ndarray, me: int, rival: int) -> int:
        own_count = np.count_nonzero(piece == me)
        rival_count = np.count_nonzero(piece == rival)
        empty = np.count_nonzero(piece == 0)

        if own_count == 4: return self.values['own_4']

        if own_count == 3 and empty == 1: return self.values['own_3']
        if own_count == 2 and empty == 2: return self.values['own_2']
        if own_count == 1 and empty == 3: return self.values['own_1']
        if rival_count == 3 and empty == 1: return self.values['rival_3']
        if rival_count == 2 and empty == 2: return self.values['rival_2']
        return 0
