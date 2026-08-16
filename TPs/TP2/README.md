# TP2: Influencia de las estrategias de muestreo

## 1. Baseline
Para la implementación de un baseline con LHS, se realiza una implementación similar a la planteada en los notebooks de la materia, sampleando de la siguiente manera:

```python
lhs = qmc.LatinHypercube(d=2)

X_pde = lhs.random(n=N_PDE)
X_pde = torch.tensor(X_pde)
```

Para el sampleo de las condiciones de borde, se mantiene la misma metodología que el TP1. La implementación completa se encuentra en [TP2_LHS.ipynb](TP2_LHS.ipynb).

## 2. RAR-D
Para la implementación del sampleo RAR-D, se construye una nueva función para entrenar la PINN en el script [tps_pinn.py](../tps_pinn.py). La implementación es la siguiente:

```python
"""
Ronda inicial usando LHS
"""
lhs = qmc.LatinHypercube(d=2)
X_pde = lhs.random(n=lhs_params['n_pde'])
X_pde = torch.tensor(X_pde)

dataset = [ torch.utils.data.TensorDataset(X_pde, torch.zeros(lhs_params['n_pde'], 1)) ] + dataset_bc
pinn, pde_loss_history, bc_vel_loss_history, bc_p_loss_history, data_loss_history, l2_error, opt = train_pinn(
    pinn, lhs_params['epochs'], dataset, device, False, # Le dejo clavado que no use DATA para entrenar
    X_data_all, y_data_all, lambdas
)

"""
Rondas de sampleo por RAR-D
"""
for round in range(rounds_params['rounds']):
    print(f"\n\nRonda {round+1} de RAR-D")
    # Obtengo nuevos puntos y los agrego al dataset de PDE
    X_winners = eval_rard(
        pinn, device,
        rounds_params['candidates'], rounds_params['winners'],
        rounds_params['k'], rounds_params['c'],
    )
    X_pde = torch.cat([X_pde.detach().to(device), X_winners.detach()], axis=0)
    dataset = [ torch.utils.data.TensorDataset(X_pde, torch.zeros(X_pde.shape[0], 1)) ] + dataset_bc
    # Continuo entrenando con los nuevos puntos
    pinn, pde_loss_history_r, bc_vel_loss_history_r, bc_p_loss_history_r, data_loss_history_r, l2_error_r, opt = train_pinn(
        pinn, rounds_params['epochs'], dataset, device, False, # Le dejo clavado que no use DATA para entrenar
        X_data_all, y_data_all, lambdas
    )
    # Guardo la evolución de las pérdidas y el error
    pde_loss_history = pde_loss_history + pde_loss_history_r
    bc_vel_loss_history = bc_vel_loss_history + bc_vel_loss_history_r
    bc_p_loss_history = bc_p_loss_history + bc_p_loss_history_r
    for i in range(3):
        l2_error[i] = l2_error[i] + l2_error_r[i]
```

En el notebook [TP2_RARD.ipynb](TP2_RARD.ipynb) se tienen los estudios realizados y sus resultados.

## 3. Resultados y discusión
Los hiperparámetros de las redes a entrenar son iguales para todas las estrategias de muestreo: 3 capas ocultas de 32 neuronas cada una, con `tanh` como función de activación.

A continuación se presentan las métricas finales de cada estrategia de muestreo.
| Tipo de sampleo | Error L2 Presión | Error L2 Velocidad U | Error L2 Velocidad V  | Tiempo de entrenamiento [s] |
|-|-|-|-|-|
| LHS                   | 0.0298 | 0.0298 | 0.0299 | 1568 |
| RAR-D (base)          | 0.0103 | 0.0097 | 0.0096 | 1497 |
| RAR-D (explotación)   | 0.0322 | 0.0276 | 0.0232 | 1492 |
| RAR-D (exploración)   | 0.0171 | 0.0176 | 0.0185 | 1509 |

En los notebooks con los estudios se pueden observar la evolución de las pérdidas y el error L2 de cada variable.

Se observa una reducción del tiempo de cómputo con RAR-D, aunque no significativa. Por otro lado, se logran mejoras importantes en las magnitudes de los errores L2 para el caso base de RAR-D, pero el caso de explotación presenta resultados comparables al baseline LHS puro. El caso de exploración presenta un rendimiento intermedio. Vale destacar que el caso de explotación presenta el mayor desbalance en el error entre variables.