# MCTry — Agente Connect-4 con MCTS

**Isabela Díaz Acosta | Fundamentos de Inteligencia Artificial 2026.1**

## Idea principal

El agente implementa **Monte Carlo Tree Search (MCTS)**, un algoritmo de mejora de política
online que construye un árbol de búsqueda usando simulaciones aleatorias (rollouts). Cada
iteración pasa por cuatro fases: selección (UCB1), expansión, simulación y retropropagación.

El parámetro clave es `budget`: el número de iteraciones MCTS por movimiento. Esto es
directamente el "límite de pensamiento" discutido en clase.

## Fundamento teórico (slides del curso)

| Componente | Slide |
|---|---|
| Estimar q(s,a) desde rollouts sin conocer P | 11 — FVMC |
| Valor en juego de suma-cero = P(ganar) | 12 — Competitive MDPs |
| UCB1 para exploración/explotación | 10 — Bandits |
| MCTS = GPI online con rollouts | 13 — Online Policy Improvement |

## Archivos

```
policy.py       ← agente final (MCTry, MCTS)
entrega.ipynb   ← análisis completo con gráficas
README.md       ← este archivo
```

## Uso

```python
from policy import MCTry

agent = MCTry(budget=300)   # 300 iteraciones por movimiento
col = agent.act(board)      # board: np.ndarray (6x7), -1=Rojo, 1=Amarillo, 0=vacío
```

### Parámetros

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `budget` | int | 300 | Iteraciones MCTS por movimiento. Más = mejor rendimiento, más tiempo |
| `c` | float | √2 | Constante de exploración UCB1 |

## Rendimiento

- Gana >97% de las partidas contra el agente aleatorio en ambos colores (budget=300).
- Tiempo promedio por movimiento: ~0.19s (budget=200), ~0.28s (budget=300).

## Versión base: Flat MC

El agente evolucionó desde una versión más simple (Flat MC) sin árbol. Ver `entrega.ipynb`
sección 3 para la comparación completa.

## Diferencias respecto a otros agentes del grupo

- No usa reglas heurísticas codificadas a mano.
- El árbol MCTS le permite "aprender" patrones durante la partida misma.
- El rendimiento mejora continuamente con más presupuesto (variable numérica continua).
