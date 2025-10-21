from mpi4py import MPI
import numpy as np
import pyamg
from tqdm import trange
from subdomain import Subdomain, calculate_gammas, calculate_A

# Global cache variables
A = None
gamma_1 = gamma_2 = gamma_3 = gamma_N = gamma_H = gamma_WF = None
lu = None
ml = None

# Global simulation parameters (to be set by main.py)
tol = maxiter = omega = n_iter = invdx = None
u_N = u_H = u_WF = dx = u_avg_guess = None
interpolation = comm = rank = None

def solve_full_MPI():
    global A, ml
    A, ml = None, None # Reset cache at the beginning of each full solve

    u1 = np.ones(invdx*(invdx-1), dtype='d')*u_avg_guess
    u2 = np.ones((2*invdx-1)*(invdx-1), dtype='d')*u_avg_guess
    u3 = np.ones(invdx*(invdx-1), dtype='d')*u_avg_guess
    u4 = np.ones(invdx//2*(invdx//2-1),dtype='d')*u_avg_guess
    bc1 = np.ones(invdx-1, dtype='d')*u_avg_guess
    bc2 = np.ones(invdx-1, dtype='d')*u_avg_guess
    bc3 = np.ones(invdx//2-1, dtype='d')*u_avg_guess
    u1o,u2o,u3o,u4o = u1.copy(),u2.copy(),u3.copy(),u4.copy()
    bc1o,bc2o,bc3o = bc1.copy(),bc2.copy(),bc3.copy()
    
    iterator = trange(n_iter, desc=f"Solving (invdx={invdx})") if rank == 1 else range(n_iter)
    for i in iterator:
        if rank==1:
            if i==0: subdomain2 = Subdomain(2, invdx, shape=(1,2), main = True)
            if i!=0:
                comm.Recv(u1,source = 0); comm.Recv(bc1,source = 0)
                comm.Recv(u3,source = 2); comm.Recv(bc2,source = 2)
                comm.Recv(u4,source = 3); comm.Recv(bc3,source = 3)
            
            u2_new,bc1_new,bc2_new,bc3_new = solve_subdomain(subdomain2,bc1,bc2,bc3)

            bc1_relaxed = bc1_new*omega + bc1o*(1-omega)
            bc2_relaxed = bc2_new*omega + bc2o*(1-omega)
            bc3_relaxed = bc3_new*omega + bc3o*(1-omega)

            comm.Send([bc1_relaxed, MPI.DOUBLE], dest =0)
            comm.Send([bc2_relaxed, MPI.DOUBLE], dest =2)
            comm.Send([bc3_relaxed, MPI.DOUBLE], dest =3)

            u1o, u3o, u4o = u1.copy(), u3.copy(), u4.copy() 
            u2o = u2_new.copy()                              
            bc1o, bc2o, bc3o = bc1_relaxed.copy(), bc2_relaxed.copy(), bc3_relaxed.copy()
            u2 = u2_new
        
        elif rank==0:
            if i==0: subdomain1 = Subdomain(1, invdx, shape=(1,1), neumann_wall='E')
            comm.Recv(bc1,source = 1)
            u1_new,bc1_new = solve_subdomain(subdomain1,bc1,bc2,bc3)
            comm.Send([u1_new, MPI.DOUBLE], dest =1); comm.Send([bc1_new, MPI.DOUBLE], dest =1)
        
        elif rank==2:
            if i==0: subdomain3 = Subdomain(3, invdx, shape=(1,1), neumann_wall='W')
            comm.Recv(bc2,source = 1)
            u3_new,bc2_new = solve_subdomain(subdomain3,bc1,bc2,bc3)
            comm.Send([u3_new, MPI.DOUBLE], dest =1); comm.Send([bc2_new, MPI.DOUBLE], dest =1)
        
        elif rank==3:
            if i == 0: subdomain4 = Subdomain(4, invdx, shape=(0.5,0.5), neumann_wall='W')
            comm.Recv(bc3,source = 1)
            u4_new,bc3_new = solve_subdomain(subdomain4,bc1,bc2,bc3)
            comm.Send([u4_new, MPI.DOUBLE], dest =1); comm.Send([bc3_new, MPI.DOUBLE], dest =1)
        
        else: break

    if rank==1:
        # After the loop, rank 1 receives the final updates from the other processes
        comm.Recv(u1,source = 0); comm.Recv(bc1,source = 0)
        comm.Recv(u3,source = 2); comm.Recv(bc2,source = 2)
        comm.Recv(u4,source = 3); comm.Recv(bc3,source = 3)

        u1 = u1*omega + u1o*(1-omega); u2 = u2*omega + u2o*(1-omega)
        u3 = u3*omega + u3o*(1-omega); u4 = u4*omega + u4o*(1-omega)

        # Assemble the final plot
        u2d = u2.reshape((2*invdx-1,1*invdx-1)) 
        u1d = u1.reshape((1*invdx-1,1*invdx))
        u3d = u3.reshape((1*invdx-1,1*invdx))
        u4d = u4.reshape((1*invdx//2-1,1*invdx//2))

        ud = np.full((2*invdx-1, 3*invdx-1), np.nan)
        ud[0:1*invdx-1, 0:1*invdx] = u1d
        ud[:, 1*invdx:2*invdx-1] = u2d
        ud[1*invdx:, 2*invdx-1:3*invdx-1] = u3d
        ud[1*invdx//2+1:1*invdx, 2*invdx-1:2*invdx+invdx//2-1] = u4d
        return ud

def solve_subdomain(subdomain,bc1,bc2,bc3):
    global gamma_1, gamma_2, gamma_3, gamma_N, gamma_H, gamma_WF, A, ml
    
    nx, ny, N = subdomain.nx, subdomain.ny, subdomain.N
    half = ny // 2
    
    if (A is None) or (A.shape[0] != N):
        A = calculate_A(subdomain)
        ml = pyamg.smoothed_aggregation_solver(A)
    
    if any(g is None for g in [gamma_1, gamma_2, gamma_3, gamma_N, gamma_H, gamma_WF]) or gamma_1.size != N:
        gamma_1, gamma_2, gamma_3, gamma_N, gamma_H, gamma_WF = calculate_gammas(subdomain)
    
    if subdomain.id==2:
        u_gamma_1 = bc1; u_gamma_2 = bc2; u_gamma_3 = bc3
        b1 = np.zeros(N, dtype='d'); b1[gamma_1==1] = u_gamma_1
        b2 = np.zeros(N, dtype='d'); b2[gamma_2==1] = u_gamma_2
        b3 = np.zeros(N, dtype='d'); b3[gamma_3==1] = u_gamma_3
        b = -(b1 + b2 + b3 + gamma_N*u_N + gamma_H*u_H + gamma_WF*u_WF)/dx**2
        u = ml.solve(b, tol=tol, maxiter=maxiter)
        
        left_col = np.arange(0, N, nx); right_col = np.arange(nx-1, N, nx)
        bc1_new = (u[left_col[:half] + 1] - b1[left_col[:half]]) / (2 * dx)
        bc2_new = (u[right_col[half+1:] - 1] - b2[right_col[half+1:]]) / (2 * dx)
        bc3_new = (u[right_col[half//2+2:half+1] - 1] - b3[right_col[half//2+2:half+1]]) / (2 * dx)
        return u,bc1_new,bc2_new,bc3_new
    
    elif subdomain.id==1:
        b1 = np.zeros(N, dtype='d'); b1[gamma_1==1] = bc1
        b = -(gamma_N*u_N + gamma_H*u_H)/dx**2 - b1/dx
        u = ml.solve(b, tol=tol, maxiter=maxiter)
        bc1_new = u[np.arange(nx-1, N, nx)]
        return u,bc1_new
    
    elif subdomain.id==3:
        b2 = np.zeros(N, dtype='d'); b2[gamma_2==1] = bc2
        b = -(gamma_N*u_N + gamma_H*u_H)/dx**2 - b2/dx
        u = ml.solve(b, tol=tol, maxiter=maxiter)
        bc2_new = u[np.arange(0, N, nx)]
        return u,bc2_new
    
    elif subdomain.id==4:
        b3 = np.zeros(N, dtype='d'); b3[gamma_3==1] = bc3
        b = -(gamma_N*u_N + gamma_H*u_H)/dx**2 - b3/dx
        u = ml.solve(b, tol=tol, maxiter=maxiter)
        bc3_new = u[np.arange(0, N, nx)]
        return u,bc3_new
