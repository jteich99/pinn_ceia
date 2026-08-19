import torch
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import qmc
import numpy as np

def set_X_bc(N_BC):
    # Dataset en puntos de condición de borde
    X_bc_u_1 = torch.hstack((torch.rand(N_BC//4, 1), torch.zeros(N_BC//4, 1)))    # Borde inferior (y=0)
    X_bc_u_2 = torch.hstack((torch.rand(N_BC//4, 1), torch.ones(N_BC//4, 1)))     # Borde superior (y=1)
    X_bc_u_3 = torch.hstack((torch.zeros(N_BC//4, 1), torch.rand(N_BC//4, 1)))    # Borde izquierdo (x=0)
    X_bc_u_4 = torch.hstack((torch.ones(N_BC//4, 1), torch.rand(N_BC//4, 1)))     # Borde derecho (x=1)
    X_bc_u = torch.vstack((X_bc_u_1, X_bc_u_2, X_bc_u_3, X_bc_u_4))

    y_bc_u_1 = torch.zeros(N_BC//4, 2)                                        # Velocidad nula
    y_bc_u_2 = torch.hstack((torch.ones(N_BC//4, 1), torch.zeros(N_BC//4, 1)))# Velocidad unitaria en x
    y_bc_u_3 = torch.zeros(N_BC//4, 2)                                        # Velocidad nula
    y_bc_u_4 = torch.zeros(N_BC//4, 2)                                        # Velocidad nula
    y_bc_u = torch.vstack((y_bc_u_1, y_bc_u_2, y_bc_u_3, y_bc_u_4))

    X_bc_p = torch.zeros(1,2)
    y_bc_p = torch.zeros(1,1)

    return X_bc_u, y_bc_u, X_bc_p, y_bc_p

def plot_points(X_pde, X_bc_u, X_bc_p, X_data=None):
    plt.scatter(X_pde[:, 0], X_pde[:, 1], s=3, color='red', label='PDE')
    plt.scatter(X_bc_u[:, 0], X_bc_u[:, 1], s=3, color='blue', label='BC vel')
    plt.scatter(X_bc_p[:, 0], X_bc_p[:, 1], s=10, color='black', marker='x', label='BC pressure')
    if X_data != None:
        plt.scatter(X_data[:, 0], X_data[:, 1], s=10, marker='^', color='green', label='DATA')

    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()

"""
Resuelve el parámetro inverso nu = 1/Re en forma cerrada, con la red congelada.

Con los pesos fijos el residuo de momento es AFÍN en nu:

    r(nu) = A - nu * B

donde A es la parte convectiva + presión y B el laplaciano. Minimizar
mean(r_x^2) + mean(r_y^2) respecto de nu es entonces un problema de mínimos
cuadrados lineal, con solución exacta

    nu* = (<A_x,B_x> + <A_y,B_y>) / (<B_x,B_x> + <B_y,B_y>)

A y B se obtienen evaluando el residuo en nu=0 y nu=1 (r(0)=A, r(1)=A-B), así
se reusa momentum_conservation y no hay que duplicar el cálculo de derivadas.
Devuelve None si el laplaciano es idénticamente nulo (problema degenerado).
"""
def solve_param_closed_form(pinn, X, device):
    Re_backup = pinn.Re
    X = X.to(device).clone().requires_grad_(True)

    y = pinn(X)

    pinn.Re = float('inf')                      # -> 1/Re = 0  =>  r = A
    A_x, A_y = pinn.momentum_conservation(X, y)
    A_x, A_y = A_x.detach(), A_y.detach()

    pinn.Re = 1.0                               # -> 1/Re = 1  =>  r = A - B
    r1_x, r1_y = pinn.momentum_conservation(X, y)
    B_x, B_y = A_x - r1_x.detach(), A_y - r1_y.detach()

    pinn.Re = Re_backup

    den = (B_x ** 2).sum() + (B_y ** 2).sum()
    if den.item() == 0.0:
        return None
    num = (A_x * B_x).sum() + (A_y * B_y).sum()
    return (num / den).item()


"""
Función para entrenamiento de objeto PINN
"""
def train_pinn(pinn, epochs, dataset, device, use_data, X_data_all, y_data_all, lambdas,
               opt=None, tags=('PDE', 'BC_VEL', 'BC_P'), param_opt=None, opt_param=None,
               epochs_no_sensor=3000,
               param_update='sgd', param_block=50, param_damping=0.3,
               param_bounds=(1e-4, 1.0), param_max_rel_step=0.1,
               stall_window=500, stall_tol=5e-3
               ):
    print("Comenzando entrenamiento del modelo...")
    
    loss = torch.nn.MSELoss()
    if opt == None:
        opt = torch.optim.LBFGS(
            pinn.parameters(), max_iter = 1,
            lr=1e-3, 
            tolerance_grad=1e-09, tolerance_change=1e-11, 
            history_size=50, line_search_fn="strong_wolfe"
        )
    epoch = 0

    # Épocas iniciales sin término de sensor ni actualización del parámetro.
    # Con un modelo ya pre-entrenado se pasa epochs_no_sensor=0 para arrancar
    # directamente en el problema inverso.
    loss_history = []
    pde_loss_history = []
    bc_vel_loss_history = []
    bc_p_loss_history = []
    data_loss_history = []
    l2_error = [[] for _ in range(3)]
    y_pred_hist = []
    param_hist = []
    grad_hist = []
    nu_star_last = float('nan')
    while epoch < epochs:
        if epoch == epochs_no_sensor:
            # Se reinicia el optimizador para evitar crasheo al cambiar de forma discontinua la función de pérdida por agregar la data de sensor
            opt.state.clear()       
        losses = {}
        def closure():
            opt.zero_grad()
            if opt_param is not None:
                # El zero_grad de LBFGS no alcanza al parámetro inverso: LBFGS llama
                # al closure varias veces por paso y el gradiente se acumularía
                opt_param.zero_grad()
            if param_opt is not None:
                # pinn.Re = torch.exp(param_opt)
                pinn.Re = 1 / param_opt
            epoch_loss = []
            for i, tag in enumerate(tags):
                X, y = dataset[i][:]
                X, y = X.to(device), y.to(device)
                if tag == 'PDE':
                    X.requires_grad = True
                    pred_y = pinn(X)
                    residue_x, residue_y = pinn.momentum_conservation(X, pred_y)
                    continuity = pinn.continuity_equation(X, pred_y)
                    loss_pde = loss(residue_x, torch.zeros_like(residue_x)) + loss(residue_y, torch.zeros_like(residue_y)) + loss(continuity, torch.zeros_like(continuity))
                    pde_loss_history.append(loss_pde.item())
                    epoch_loss.append(lambdas['pde'] * loss_pde)
                    losses['PDE'] = loss_pde.item()
                elif tag == 'BC_VEL':
                    # y_bc_u ya viene como (N,2) = (u, v): no hay que slicearlo
                    y_pred = pinn(X)[:, 1:3]
                    loss_bc = loss(y_pred, y)
                    bc_vel_loss_history.append(loss_bc.item())
                    epoch_loss.append(lambdas['bc_u'] * loss_bc)
                    losses['BC_VEL'] = loss_bc.item()
                elif tag == 'BC_P':
                    y = y[:, 0]
                    y_pred = pinn(X)[:, 0]
                    loss_bc_p = loss(y_pred, y)
                    bc_p_loss_history.append(loss_bc_p.item())
                    epoch_loss.append(lambdas['bc_p'] * loss_bc_p)
                    losses['BC_P'] = loss_bc_p.item()
                elif tag == 'DATA' and use_data:
                    y_pred = pinn(X)
                    loss_data = loss(y_pred, y)
                    data_loss_history.append(loss_data.item())
                    epoch_loss.append(loss_data)
                    losses['DATA'] = loss_data.item()
                elif tag == 'SENSOR':
                    if epoch >= epochs_no_sensor:
                        y_pred = pinn(X)[:,1:]
                        loss_data = loss(y_pred, y)
                        data_loss_history.append(loss_data.item())
                        epoch_loss.append(lambdas["sensor"] * lambdas['pde'] * loss_data)
                        losses[tag] = loss_data.item()
                    else:
                        losses[tag] = 0


            epoch_loss = sum(epoch_loss)
            epoch_loss.backward()
            return epoch_loss

        opt.step(closure)

        # Actualización del parámetro inverso, después del paso de LBFGS sobre la red
        if param_opt is not None and epoch >= epochs_no_sensor:
            if param_update == 'closed_form':
                # Alternancia por bloques: cada param_block épocas se congela la
                # red y se resuelve nu exacto por mínimos cuadrados. Al no ser un
                # paso de gradiente no hay lr que ajustar ni forma de irse a un
                # borde: nu* siempre cae donde el campo actual lo pide.
                if (epoch - epochs_no_sensor) % param_block == 0:
                    nu_star = solve_param_closed_form(pinn, dataset[0][:][0], device)
                    if nu_star is not None:
                        nu_star_last = nu_star
                    # nu* <= 0 es no físico (viscosidad negativa): pasa cuando el
                    # campo todavía está lejos de una solución de NS. En ese caso
                    # se deja nu como está en vez de empujarlo contra el clamp.
                    if nu_star is not None and nu_star > 0:
                        with torch.no_grad():
                            nu_old = param_opt.item()
                            # Amortiguado: la red fue ajustada al nu viejo, saltar
                            # directo a nu* hace oscilar la alternancia
                            param_opt.fill_(nu_old + param_damping * (nu_star - nu_old))
                            param_opt.clamp_(min=param_bounds[0], max=param_bounds[1])
                grad_hist.append(float('nan'))
            elif opt_param is not None:
                if (epoch - epochs_no_sensor) % param_block == 0:
                    # closure() ya hace zero_grad y backward por dentro: volver a llamar
                    # a backward sobre el loss devuelto recorre un grafo ya liberado
                    closure()
                    # Gradiente que consume opt_param, ya en el punto aceptado.
                    # Con SGD el paso es lr * grad, así que la magnitud de este número
                    # define directamente cuánto se mueve el parámetro por época.
                    grad_hist.append(
                        float(param_opt.grad.detach().cpu()) if param_opt.grad is not None else float('nan')
                    )
                    # Trust region relativo: el paso de SGD es lr*|g|, y con nu en
                    # escala 1e-2 y gradientes O(0.1-1) el paso puede ser mayor que
                    # el propio nu -> cruza el mínimo cada época y el parámetro
                    # cicla entre los clamps. Se recorta el gradiente para que
                    # |delta nu| por época no supere param_max_rel_step * nu, con lo
                    # cual el paso queda proporcional al valor actual del parámetro
                    # (mismo efecto estabilizador que optimizar en log, sin cambiar
                    # la variable optimizada).
                    if param_opt.grad is not None:
                        with torch.no_grad():
                            lr_eff = opt_param.param_groups[0]['lr']
                            g_max = param_max_rel_step * float(param_opt.abs()) / lr_eff
                            param_opt.grad.clamp_(min=-g_max, max=g_max)
                    opt_param.step()
                    with torch.no_grad():
                        param_opt.clamp_(min=param_bounds[0], max=param_bounds[1])
                    # Re* como diagnóstico también en modo sgd: nu* en forma
                    # cerrada sobre el campo actual, cada param_block épocas.
                    # No actualiza el parámetro; sirve para chequear que SGD
                    # converge al mismo mínimo que pide el campo.
                # if (epoch - epochs_no_sensor) % param_block == 0:
                #     nu_star = solve_param_closed_form(pinn, dataset[0][:][0], device)
                #     if nu_star is not None:
                #         nu_star_last = nu_star
            else:
                grad_hist.append(float('nan'))
        elif param_opt is not None:
            grad_hist.append(float('nan'))

        if param_opt is not None:
            # param_hist.append(float(torch.exp(param_opt).detach().cpu()))
            param_hist.append(float((1 / param_opt).detach().cpu()))

        # Corte por convergencia: la zona de spikes (plateau de pérdidas con
        # transitorios de 1 época por bloque) más Re ~constante es el punto en
        # que el objetivo ya no distingue valores de Re -> seguir entrenando
        # solo hace derivar el par (red, nu) por el valle plano. Se exigen las
        # DOS condiciones: pérdidas planas solas pueden ser deriva de valle
        # (Re todavía se mueve), y Re plano solo puede ser una pausa transitoria.
        if (param_opt is not None and stall_window is not None
                and epoch >= epochs_no_sensor
                and len(param_hist) >= 2 * stall_window
                and (epoch - epochs_no_sensor) % param_block == 0):
            W = stall_window
            # Re también se compara por medianas de ventana: en la zona de
            # spikes Re suele ciclar entre 2-4 valores fijos (ciclo límite de la
            # alternancia red<->nu), y comparar punto a punto puede caer en
            # fases distintas del ciclo y ver "cambio" para siempre. La mediana
            # de un ciclo estacionario es constante -> corta; si hay deriva
            # real por el valle, la mediana se mueve -> sigue entrenando.
            re_now = np.median(param_hist[-W:])
            re_then = np.median(param_hist[-2 * W:-W])
            re_change = abs(re_now - re_then) / abs(re_then)
            # pde_loss_history tiene varias entradas por época (evaluaciones de
            # línea de búsqueda de LBFGS + closure extra por bloque): se escala
            # la ventana de épocas a entradas con el ratio observado. Medianas
            # para que los spikes de 1 época no muevan la métrica.
            k = max(1, len(pde_loss_history) // max(1, len(param_hist)))
            Wl = W * k
            if len(pde_loss_history) >= 2 * Wl:
                lo_now = np.median(pde_loss_history[-Wl:])
                lo_then = np.median(pde_loss_history[-2 * Wl:-Wl])
                loss_change = abs(lo_now - lo_then) / abs(lo_then)
                if re_change < stall_tol and loss_change < stall_tol:
                    print(f"\nCorte por convergencia en epoch {epoch}: "
                          f"Re={re_now:.1f} (mediana; cambio rel. {re_change:.1e} "
                          f"entre ventanas de {W} épocas), PDE loss en plateau "
                          f"(cambio rel. {loss_change:.1e})")
                    break

        # Calculo la norma-2 del error
        pinn.eval()
        with torch.no_grad():
            y_pred = pinn(torch.tensor(X_data_all, dtype=torch.float64).to(device)).detach().cpu().numpy()
            for i in range(3):
                l2_error[i].append(linalg.norm(y_data_all.numpy()[:,i] - y_pred[:,i]))

        # if epoch % 10 == 0:
        if epoch % 1 == 0:
            # print(f'Epoch {epoch}, PDE Loss: {losses["PDE"]:.6f}, BC vel. Loss: {losses["BC_VEL"]:.6f}, BC pres. Loss: {losses["BC_P"]:.6f}, DATA Loss: {losses["DATA"] if use_data else "N/A"}, L2 Error: {l2_error[-1]:.6f}')
            if param_opt is not None:
                print(f'\rEpoch {epoch}, PDE Loss: {losses["PDE"]:.6f}, BC vel. Loss: {losses["BC_VEL"]:.6f}, BC pres. Loss: {losses["BC_P"]:.6f}, DATA Loss: {losses["DATA"] if use_data else "N/A"}, SENSOR Loss: {losses["SENSOR"]:.6f}, Re: {param_hist[-1]:.1f}, grad: {grad_hist[-1]:+.3e}, Re*: {1/nu_star_last if nu_star_last == nu_star_last else float("nan"):.1f}', end="", flush=True)
            else:
                print(f'\rEpoch {epoch}, PDE Loss: {losses["PDE"]:.6f}, BC vel. Loss: {losses["BC_VEL"]:.6f}, BC pres. Loss: {losses["BC_P"]:.6f}, DATA Loss: {losses["DATA"] if use_data else "N/A"}', end="", flush=True)

        pinn.train()

        # guardado de y_pred
        if epoch % 500 == 0:
            y_pred_hist.append((epoch, y_pred))

        epoch += 1

    if param_opt is not None:
        return pinn, pde_loss_history, bc_vel_loss_history, bc_p_loss_history, data_loss_history, l2_error, opt, param_hist, grad_hist
    else:
        return pinn, pde_loss_history, bc_vel_loss_history, bc_p_loss_history, data_loss_history, l2_error, opt


"""
Entrenamiento utilizando estrategia RAR-D para sampleo de puntos de colocación de la PINN
"""
def train_pinn_rard(
    pinn,
    dataset_bc,
    lhs_params, rounds_params,
    device, lambdas,
    X_data_all, y_data_all
):
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

    return pinn, pde_loss_history, bc_vel_loss_history, bc_p_loss_history, data_loss_history, l2_error, opt

def train_pinn_inverse(
    pinn, 
    param,
    epochs, 
    dataset, 
    device, 
    X_data_all, 
    y_data_all, 
    lambdas,
    tags,
    epochs_no_sensor=3000,
    lr_param = 1e-3,
    param_update='sgd', param_block=50, param_damping=0.3,
    param_bounds=(1e-4, 1.0), param_max_rel_step=0.1,
    stall_window=500, stall_tol=5e-3
):
    if not isinstance(param, torch.nn.Parameter):
        param = torch.nn.Parameter(torch.as_tensor(param, dtype=torch.float64, device=device))
    # pinn.log_Re = param
    # pinn.Re = torch.exp(pinn.log_Re)
    pinn.inv_Re = param
    pinn.Re = (1 / pinn.inv_Re)

    # La red va con LBFGS; el parámetro inverso queda excluido de este optimizador
    net_params = [p for p in pinn.parameters() if p is not param]
    opt = torch.optim.LBFGS(
        net_params,
        max_iter = 1, lr=1e-3,
        tolerance_grad=1e-07, tolerance_change=1e-11,
        history_size=50, line_search_fn="strong_wolfe"
    )
    # El parámetro inverso se optimiza aparte.
    # Adam es invariante de escala (se comporta como signSGD): avanza ~lr por época
    # sin importar cuán chico sea el gradiente, así que no frena al acercarse al
    # mínimo y se va de largo hasta el clamp. SGD conserva la magnitud del
    # gradiente, con lo cual el paso se achica solo y hay un punto fijo real en g=0.
    # OJO: lr_param debe ser tal que lr*|g| << nu (nu ~ 1e-2, g ~ O(0.1-1)):
    # con lr=1e-3 el paso libre es ~1e-4 por época, y el trust region relativo
    # (param_max_rel_step, ver train_pinn) acota además cada paso al 10% del nu
    # actual, que es lo que evita el ciclo entre clamps que daba lr=0.1.
    opt_param = torch.optim.SGD([param], lr=lr_param)
    # opt_param = torch.optim.Adam([param], lr=lr_param, eps=1e-16)

    pinn, pde_loss_history, bc_vel_loss_history, bc_p_loss_history, data_loss_history, l2_error, opt, param_hist, grad_hist = train_pinn(
            pinn, epochs, dataset, device, False, # Le dejo clavado que no use DATA para entrenar
            X_data_all, y_data_all, lambdas,
            opt=opt, tags=tags, param_opt = param, opt_param = opt_param,
            epochs_no_sensor = epochs_no_sensor,
            param_update = param_update, param_block = param_block,
            param_damping = param_damping, param_bounds = param_bounds,
            param_max_rel_step = param_max_rel_step,
            stall_window = stall_window, stall_tol = stall_tol
        )

    return pinn, pde_loss_history, bc_vel_loss_history, bc_p_loss_history, data_loss_history, l2_error, opt, param_hist, grad_hist



def eval_rard(
    pinn, device,
    candidates, winners,
    k, c,
):
    lhs = qmc.LatinHypercube(d=2)
    X_candidates = torch.tensor(lhs.random(n=candidates)).to(device)
    X_candidates.requires_grad = True
    y_pred_candidates = pinn(X_candidates)
    # Evaluo las ecuaciones de conservación
    r_u, r_v = pinn.momentum_conservation(X_candidates, y_pred_candidates)
    r_c = pinn.continuity_equation(X_candidates, y_pred_candidates)
    # Calculo métrica y densida de muestreo
    eps = torch.sqrt(torch.pow(r_u,2) + torch.pow(r_v,2) + torch.pow(r_c,2))
    p_muestreo = torch.pow(eps, k) / torch.pow(eps, k).mean() + c
    p_muestreo_norm = (p_muestreo/sum(p_muestreo))[:]
    # Obtengo los puntos ganadores
    idx = np.random.choice(a=len(X_candidates), size=winners, replace=False, p=p_muestreo_norm.data.detach().cpu().numpy())
    X_winners = X_candidates[idx.flatten(), :]
    return X_winners


def plot_hist(loss_history, pde_loss_history, bc_vel_loss_history, bc_p_loss_history, l2_error, use_data = False, data_loss_history = []):
    axs = 6 if use_data else 5
    fig, axs = plt.subplots(1, axs, figsize=(5*axs, 5))
    axs[0].plot(loss_history, label='Total Loss')
    axs[0].set_title('Total Loss')
    axs[1].plot(pde_loss_history, label='PDE Loss')
    axs[1].set_title('PDE Loss')
    axs[2].plot(bc_vel_loss_history, label='BC Velocity Loss')
    axs[2].set_title('BC Velocity Loss')
    axs[3].plot(bc_p_loss_history, label='BC Pressure Loss')
    axs[3].set_title('BC Pressure Loss')
    if use_data:
        axs[4].plot(data_loss_history, label='DATA Loss')
        axs[4].set_title('DATA Loss')
    for i, var in zip(range(3), ('p', 'u', 'v')):
        axs[5 if use_data else 4].plot(l2_error[i], label=f'L2 Error {var}')
    axs[5 if use_data else 4].set_title('L2 Error')
    [ax.set_yscale('log') for ax in axs]
    [ax.set_xscale('log') for ax in axs]
    plt.show()

"""
Clase PINN
"""
class PINN(torch.nn.Module):
    def __init__(self, model_params):
        super(PINN, self).__init__()
        self.hidden_layers = model_params['hidden_layers']
        self.hidden_units = model_params['hidden_units']
        self.activation = model_params['activation']
        self.Re = model_params['Re']
        self.device = model_params['device']

        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, self.hidden_units)] +
            [torch.nn.Linear(self.hidden_units, self.hidden_units) for _ in range(self.hidden_layers - 1)] +
            [torch.nn.Linear(self.hidden_units, 3)]
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == torch.nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = torch.nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            m.bias.data.fill_(0)

    def forward(self, x):
        y = x
        for layer in self.layers[:-1]:
            y = self.activation(layer(y))
        y = self.layers[-1](y)
        return y

    def momentum_conservation(self, x, y):
        p = y[:, 0]
        u = y[:, 1]
        v = y[:, 2]

        p_grad = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_x, p_y = p_grad[:, 0], p_grad[:, 1]
        u_grad = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x, u_y = u_grad[:, 0], u_grad[:, 1]
        v_grad = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x, v_y = v_grad[:, 0], v_grad[:, 1]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:, 0]
        v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:, 1]

        # momentum equations (Navier-Stokes) for incompressible flow
        residue_x = u * u_x + v * u_y + p_x - (1/self.Re) * (u_xx + u_yy)
        residue_y = u * v_x + v * v_y + p_y - (1/self.Re) * (v_xx + v_yy)
        return residue_x, residue_y 

    def continuity_equation(self, x, y):
        u = y[:, 1]
        v = y[:, 2]

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]
        v_y = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, 1]

        return u_x + v_y

