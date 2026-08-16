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
Función para entrenamiento de objeto PINN
"""
def train_pinn(pinn, epochs, dataset, device, use_data, X_data_all, y_data_all, lambdas):
    print("Comenzando entrenamiento del modelo...")
    tags = ['PDE', 'BC_VEL', 'BC_P']
    loss = torch.nn.MSELoss()
    opt = torch.optim.LBFGS(
        pinn.parameters(), max_iter = 1,
        lr=1e-3, 
        tolerance_grad=1e-09, tolerance_change=1e-11, 
        history_size=50, line_search_fn="strong_wolfe"
    )
    epoch = 0


    loss_history = []
    pde_loss_history = []
    bc_vel_loss_history = []
    bc_p_loss_history = []
    data_loss_history = []
    l2_error = [[] for _ in range(3)]
    y_pred_hist = []
    while epoch < epochs:
        losses = {}
        def closure():
            opt.zero_grad()
            epoch_loss = []
            for tag in tags:
                if tag == 'PDE':
                    X, y = dataset[0][:]
                    X, y = X.to(device), y.to(device)
                    X.requires_grad = True
                    pred_y = pinn(X)
                    residue_x, residue_y = pinn.momentum_conservation(X, pred_y)
                    continuity = pinn.continuity_equation(X, pred_y)
                    loss_pde = loss(residue_x, torch.zeros_like(residue_x)) + loss(residue_y, torch.zeros_like(residue_y)) + loss(continuity, torch.zeros_like(continuity))
                    pde_loss_history.append(loss_pde.item())
                    epoch_loss.append(lambdas['pde'] * loss_pde)
                    losses['PDE'] = loss_pde.item()
                elif tag == 'BC_VEL':
                    X, y_vel = dataset[1][:]
                    X, y_vel = X.to(device), y_vel.to(device)
                    # X.requires_grad = True
                    y_pred = pinn(X)[:, 1:3]
                    loss_bc = loss(y_pred, y_vel)
                    bc_vel_loss_history.append(loss_bc.item())
                    epoch_loss.append(lambdas['bc_u'] * loss_bc)
                    losses['BC_VEL'] = loss_bc.item()
                elif tag == 'BC_P':
                    X, y_p = dataset[2][:]
                    X, y_p = X.to(device), y_p.to(device)
                    # X.requires_grad = True
                    y_pred = pinn(X)[:, 0:1]
                    loss_bc_p = loss(y_pred, y_p)
                    bc_p_loss_history.append(loss_bc_p.item())
                    epoch_loss.append(lambdas['bc_p'] * loss_bc_p)
                    losses['BC_P'] = loss_bc_p.item()
                elif tag == 'DATA' and use_data:
                    X, y_data = dataset[3][:]
                    X, y_data = X.to(device), y_data.to(device)
                    y_pred = pinn(X)
                    loss_data = loss(y_pred, y_data)
                    data_loss_history.append(loss_data.item())
                    epoch_loss.append(loss_data)
                    losses['DATA'] = loss_data.item()

            epoch_loss = sum(epoch_loss)
            epoch_loss.backward()
            return epoch_loss

        opt.step(closure)        

        # Calculo la norma-2 del error
        pinn.eval()
        with torch.no_grad():
            y_pred = pinn(torch.tensor(X_data_all, dtype=torch.float64).to(device)).detach().cpu().numpy()
            for i in range(3):
                l2_error[i].append(linalg.norm(y_data_all.numpy()[i,:] - y_pred[i,:]))

        if epoch % 10 == 0:
            # print(f'Epoch {epoch}, PDE Loss: {losses["PDE"]:.6f}, BC vel. Loss: {losses["BC_VEL"]:.6f}, BC pres. Loss: {losses["BC_P"]:.6f}, DATA Loss: {losses["DATA"] if use_data else "N/A"}, L2 Error: {l2_error[-1]:.6f}')
            print(f'Epoch {epoch}, PDE Loss: {losses["PDE"]:.6f}, BC vel. Loss: {losses["BC_VEL"]:.6f}, BC pres. Loss: {losses["BC_P"]:.6f}, DATA Loss: {losses["DATA"] if use_data else "N/A"}')

        # guardado de y_pred
        if epoch % 500 == 0:
            y_pred_hist.append((epoch, y_pred))

        epoch += 1

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

