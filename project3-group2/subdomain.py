import numpy as np
from scipy.sparse import diags
from dataclasses import dataclass

@dataclass
class Subdomain:
    id: int
    invdx: int
    shape: tuple = (1, 1) # x, y
    neumann_wall: str = None
    main: bool = None
    
    def __post_init__(self):
        self.nx = int(self.shape[0]*self.invdx)
        self.ny = int(self.shape[1]*self.invdx)
        if self.main == True or self.neumann_wall in ['N', 'S']:
            self.nx -= 1
        if self.neumann_wall in ['W', 'E', None]:
            self.ny -= 1
        self.N = self.nx*self.ny

    def get_neumann_BC_indices(self):
        if self.neumann_wall == 'W':
            neumann_indices = np.arange(0, self.N, self.nx)
        elif self.neumann_wall == 'E':
            neumann_indices = np.arange(self.nx-1, self.N, self.nx)
        elif self.neumann_wall == 'N':
            neumann_indices = np.arange(0, self.nx)
        elif self.neumann_wall == 'S':
            neumann_indices = np.arange(self.N-self.nx, self.N)
        else:
            return None
        return neumann_indices

def calculate_gammas(subdomain):
    nx = subdomain.nx
    ny = subdomain.ny
    N = subdomain.N
    half = ny // 2
    
    gamma_N = np.zeros(N, dtype=int)
    gamma_H = np.zeros(N, dtype=int)
    gamma_WF = np.zeros(N, dtype=int)
    gamma_1 = np.zeros(N, dtype=int)
    gamma_3 = np.zeros(N, dtype=int)
    gamma_2 = np.zeros(N, dtype=int)
    
    left_col = np.arange(0, N, nx)
    right_col = np.arange(nx-1, N, nx)
    
    if subdomain.id==1:
        gamma_H[left_col[:]] = 1
        gamma_1[right_col[:]] = 1
        top_row = np.arange(N - nx, N)
        gamma_N[top_row] = 1
        bot_row = np.arange(nx)
        gamma_N[bot_row] = 1
    elif subdomain.id==2:
        gamma_1[left_col[:half]] = 1
        gamma_N[left_col[half:]] = 1
        gamma_N[right_col[:half//2+2]] = 1
        gamma_2[right_col[half+1:]] = 1
        gamma_3[right_col[half//2+2:half+1]] = 1
        if ny%2==1:
            gamma_2[right_col[half]] = 0
        top_row = np.arange(N - nx, N)
        gamma_H[top_row] = 1
        bot_row = np.arange(nx)
        gamma_WF[bot_row] = 1
    elif subdomain.id == 3:
        gamma_2[left_col[:]] = 1
        gamma_H[right_col[:]] = 1
        top_row = np.arange(N - nx, N)
        gamma_N[top_row] = 1
        bot_row = np.arange(nx)
        gamma_N[bot_row] = 1
    elif subdomain.id==4:
        gamma_3[left_col[:]] = 1
        gamma_N[right_col[:]] = 1
        top_row = np.arange(N - nx, N)
        gamma_N[top_row] += 1
        bot_row = np.arange(nx)
        gamma_H[bot_row] = 1
        
    return gamma_1, gamma_2, gamma_3, gamma_N, gamma_H, gamma_WF

def calculate_A(subdomain):
    size = subdomain.N
    nx = subdomain.nx
    center = -4 * np.ones(size)
    
    right = np.ones(size-1)
    mask_indices_r = np.arange(nx-1, size-1, nx)
    right[mask_indices_r] = 0
    
    left = np.ones(size-1)
    mask_indices_l = np.arange(nx, size, nx)
    left[mask_indices_l-1] = 0
    
    up = np.ones(size - nx)
    down = np.ones(size - nx)
    
    neumann_BC_indices = subdomain.get_neumann_BC_indices()
    if neumann_BC_indices is not None:
        center[neumann_BC_indices] = -3
        
    diagonals = [center, left, right, down, up]
    offsets = [0, -1, 1, -nx, nx]
    
    A = diags(diagonals, offsets, shape=(size, size), format='csr')
    
    dx = 1 / subdomain.invdx
    return A / (dx**2)