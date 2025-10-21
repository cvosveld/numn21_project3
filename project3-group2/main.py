from mpi4py import MPI
import matplotlib.pyplot as plt
from simulation import solve_full_MPI
import simulation 

def plot_apt(u, title='', interpolation='none'):
    """Displays the temperature distribution of the apartment."""
    plt.imshow(u, origin='lower', cmap='jet', interpolation=interpolation, vmin=5, vmax=40)
    plt.colorbar(label='Temperature (u)')
    plt.title(title)
    plt.show()

class Setup:
    """A class to hold simulation parameters and run the simulation."""
    def __init__(self, tol=1e-4, maxiter=100, omega=0.05, n_iter=10, invdx=100, u_N=15, u_H=40, u_WF=5):
        self.params = {
            'tol': tol, 'maxiter': maxiter, 'omega': omega,
            'n_iter': n_iter, 'invdx': invdx, 'u_N': u_N,
            'u_H': u_H, 'u_WF': u_WF
        }

    def get_solution(self):
        """Sets global parameters in the simulation module and runs it."""
        for key, value in self.params.items():
            setattr(simulation, key, value)
        
        simulation.dx = 1 / self.params['invdx']
        simulation.u_avg_guess = (6*self.params['u_N'] + 2.5*self.params['u_H'] + 1*self.params['u_WF']) / 9.5
        simulation.interpolation = interpolation
        simulation.comm = comm
        simulation.rank = rank
        
        return solve_full_MPI()

# --- Main execution block ---
if __name__ == "__main__":
    interpolation = 'none' # bilinear, bicubic, none

    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    if rank == 0:
        print("Starting 4-room simulation with MPI...")

    # Define a standard setup
    setup1 = Setup(tol=1e-4, maxiter=100, omega=0.8, n_iter=20, invdx=100, u_N=15, u_H=40, u_WF=5)
    
    # Define a setup with a colder heater to test parameter changes
    setup2 = Setup(tol=1e-4, maxiter=100, omega=0.8, n_iter=20, invdx=100, u_N=15, u_H=10, u_WF=5)

    # --- Run simulations ---
    ud1 = setup1.get_solution()
    
    # Reset cache for the second run. 
    simulation.A = None 
    ud2 = setup2.get_solution()

    # --- Plot results on the coordinating rank ---
    if rank == 1:
        plot_apt(ud1, title=f'4-Room Solution (Heater at {setup1.params["u_H"]}C)', interpolation=interpolation)
        plot_apt(ud2, title=f'4-Room Solution (Heater at {setup2.params["u_H"]}C)', interpolation=interpolation)
