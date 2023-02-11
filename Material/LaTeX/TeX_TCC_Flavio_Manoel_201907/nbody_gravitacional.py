# Bibliotecas importadas
import math
import numpy as np
from timeit import default_timer as timer
import numba as nb
from numba import jit, njit, prange, cuda 

#Definições
G = 6.67408*1e-11 #define a constante gravitacional
sft = 1e-10 #constante de atenuamento para tratamento das singularidades

#------ Funções de simulação ------#

## Função para Python puro e Numba CPU serial
@njit #decorador do compilador Numba
def gravitationalNbody_NumbaCPU(r, v, a, m, nobj, ninter, dt, 
                                flag_Energy=False):
    for inter in range(ninter):
        for i in range(nobj):
            a[i][0] = 0.0; a[i][1] = 0.0; a[i][2] = 0.0
            for j in range(nobj):
                if i != j:
                    # calculate a_ij acceleration
                    dx = r[j][0] - r[i][0]
                    dy = r[j][1] - r[i][1]
                    dz = r[j][2] - r[i][2]
                    dsq = dx*dx + dy*dy + dz*dz + sft*sft
                    gA = (G*m[j])/(dsq*math.sqrt(dsq))
                    # sum a_ij_x to a_i_x, etc
                    a[i][0] += gA*dx
                    a[i][1] += gA*dy
                    a[i][2] += gA*dz
            # update the v_i velocity
            v[i][0] += a[i][0]*dt
            v[i][1] += a[i][1]*dt
            v[i][2] += a[i][2]*dt
        for i in range(nobj):
            # update the r_i position
            r[i][0] += v[i][0]*dt
            r[i][1] += v[i][1]*dt
            r[i][2] += v[i][2]*dt
    K = 0.0; U = 0.0
    if flag_Energy:
        for i in range(nobj):
            # Sum the total kinectic energy
            K += 0.5*m[i]*(v[i][0]**2 + v[i][1]**2 + v[i][2]**2)
            for j in range(nobj):
                if (i < j):
                    # Sum the total potential energy
                    U += (-G*m[i]*m[j])/math.sqrt((r[j][0] - r[i][0])**2 + 
                                                  (r[j][1] - r[i][1])**2 + 
                                                  (r[j][2] - r[i][2])**2 + 
                                                  sft**2)
    return K, U


## Função para Python com Numba CPU em paralelo
@njit(parallel=True) #decorador do compilador Numba para paralelização em CPU
def gravitationalNbody_NumbaCPUparallel(r, v, a, m, nobj, ninter, dt, 
                                        flag_Energy=False):
    for inter in range(ninter):
        for i in prange(nobj):
            a[i][0] = 0.0; a[i][1] = 0.0; a[i][2] = 0.0
            for j in range(nobj):
                if i != j:
                    # calculate a_ij acceleration
                    dx = r[j][0] - r[i][0]
                    dy = r[j][1] - r[i][1]
                    dz = r[j][2] - r[i][2]
                    dsq = dx*dx + dy*dy + dz*dz + sft*sft
                    gA = (G*m[j])/(dsq*math.sqrt(dsq))
                    # sum a_ij_x to a_i_x, etc
                    a[i][0] += gA*dx
                    a[i][1] += gA*dy
                    a[i][2] += gA*dz
            # update the v_i velocity
            v[i][0] += a[i][0]*dt
            v[i][1] += a[i][1]*dt
            v[i][2] += a[i][2]*dt
        for i in prange(nobj):
            # update the r_i position
            r[i][0] += v[i][0]*dt
            r[i][1] += v[i][1]*dt
            r[i][2] += v[i][2]*dt
    K = 0.0; U = 0.0
    if flag_Energy:
        for i in prange(nobj):
            # Sum the total kinectic energy
            K += 0.5*m[i]*(v[i][0]**2 + v[i][1]**2 + v[i][2]**2)
            for j in range(nobj):
                if (i < j):
                    # Sum the total potential energy
                    U += (-G*m[i]*m[j])/math.sqrt((r[j][0] - r[i][0])**2 + 
                                                  (r[j][1] - r[i][1])**2 + 
                                                  (r[j][2] - r[i][2])**2 + 
                                                  sft**2)
    return K, U


## Função para Python com Numpy em CPU serial
@njit #decorador do compilador Numba
def gravitationalNbody_NumPy_NumbaCPU(r, v, a, m, nobj, ninter, dt, 
                                      flag_Energy=False):
    for inter in range(ninter):
        # zero the a acceleration
        a = np.zeros((nobj, 3))
        for i in range(nobj):
            # update the a acceleration
            dr = r[i] - r
            dsq = np.sum(dr*dr, axis=1) + sft*sft
            gA = (G*m[i])/(dsq*np.sqrt(dsq))
            a += (dr.T*gA).T
        # update the v velocity
        v += a*dt
        # update the r position
        r += v*dt
    K = 0.0; U = 0.0
    if flag_Energy:
        # Sum the total kinetic energy
        K = np.sum(0.5*m*np.sum(v*v, axis=1))
        U = 0.0
        for i in range(nobj):
            for j in range(nobj):
                if (i < j):
                    # Sum the total potential energy
                    U += (-G*m[i]*m[j])/math.sqrt((r[j][0] - r[i][0])**2 + 
                                                  (r[j][1] - r[i][1])**2 + 
                                                  (r[j][2] - r[i][2])**2 + 
                                                  sft**2)
    return K, U

## Função para Python com Numpy e Numba CPU em paralelo
@njit(parallel=True) #decorador do compilador Numba para paralelização em CPU
def gravitationalNbody_NumPy_NumbaCPUparallel(r, v, a, m, nobj, ninter, dt, 
                                              flag_Energy=False):
    for inter in range(ninter):
        # zero the a acceleration
        a = np.zeros((nobj, 3))
        for i in prange(nobj):
            # update the a acceleration
            dr = r[i] - r
            dsq = np.sum(dr*dr, axis=1) + sft*sft
            gA = (G*m[i])/(dsq*np.sqrt(dsq))
            a += (dr.T*gA).T
        # update the v velocity
        v += a*dt
        # update the r position
        r += v*dt
    K = 0.0; U = 0.0
    if flag_Energy:
        K = np.sum(0.5*m*np.sum(v*v, axis=1))
        U = 0.0
        for i in prange(nobj):
            for j in range(nobj):
                if (i < j):
                    # Sum the total potential energy
                    U += (-G*m[i]*m[j])/math.sqrt((r[j][0] - r[i][0])**2 + 
                                                  (r[j][1] - r[i][1])**2 + 
                                                  (r[j][2] - r[i][2])**2 + 
                                                  sft**2)
    return K, U

### Funções para Python com Numba paralelizado em GPU via CUDA
# Cálculo da Aceleração e atualização das velocidades
@cuda.jit("(float64[:,:], float64[:,:], float64[:,:], float64[:], int32, float64)")
#cuda.jit é o decorador do compilador Numba para paralelismo em GPU
def gravitationalNbody_a_v_ij_NumbaGPU64(r, v, a, m, nobj, dt):
    i = cuda.grid(1)
    if i < nobj:
        a[i][0] = 0.0; a[i][1] = 0.0; a[i][2] = 0.0
        for j in range(nobj):
            if i != j:
                dx = r[j][0] - r[i][0]
                dy = r[j][1] - r[i][1]
                dz = r[j][2] - r[i][2]
                dsq = dx*dx + dy*dy + dz*dz + sft*sft
                gA = (G*m[j])/(dsq*math.sqrt(dsq))
                # sum a_ij_x to a_i_x, etc
                a[i][0] += gA*dx
                a[i][1] += gA*dy
                a[i][2] += gA*dz
        v[i][0] += a[i][0]*dt
        v[i][1] += a[i][1]*dt
        v[i][2] += a[i][2]*dt

# Atualização das posições
@cuda.jit("(float64[:,:], float64[:,:], int32, float64)")        
def gravitationalNbody_r_i_NumbaGPU64(r, v, nobj, dt):
    i = cuda.grid(1)
    if i < nobj:
        r[i][0] += v[i][0]*dt
        r[i][1] += v[i][1]*dt
        r[i][2] += v[i][2]*dt

# Cálculo da Energia total
@cuda.jit("(float64[:,:], float64[:,:], float64[:], int32, float64[:])")
def gravitationalNbody_K_U_NumbaCUDA(r, v, m, nobj, K_U):
    i = cuda.grid(1)
    if i < nobj:
        K_U[0] += 0.5*m[i]*(v[i][0]**2 + v[i][1]**2 + v[i][2]**2)
        for j in range(nobj):
            if (i < j):
                K_U[1] += (-G*m[i]*m[j])/math.sqrt((r[j][0] - r[i][0])**2 + 
                                                   (r[j][1] - r[i][1])**2 + 
                                                   (r[j][2] - r[i][2])**2 +
                                                   sft**2)

# Chamada das funções de paralelismo em GPU
@jit(parallel=True)
def gravitationalNbody_NumbaGPU64(r, v, a, m, nobj, ninter, dt, flag_Energy=False):
    threadsperblock = 128
    blockspergrid = (nobj + (threadsperblock - 1)) // threadsperblock
    print("Threads per block = {}; Blocks per grid = {}".format(threadsperblock, 
                                                                blockspergrid))
    htod_ti = timer()
    d_r = cuda.to_device(r)
    d_v = cuda.to_device(v)
    d_a = cuda.to_device(a)
    d_m = cuda.to_device(m)
    htod_tf = timer()
    for inter in range(ninter):
        gravitationalNbody_a_v_ij_NumbaGPU64[blockspergrid, 
                                             threadsperblock](d_r, d_v, d_a,
                                                              d_m, nobj, dt)
        cuda.synchronize()
        gravitationalNbody_r_i_NumbaGPU64[blockspergrid, 
                                          threadsperblock](d_r, d_v, nobj, dt)
        cuda.synchronize()
    dtoh_ti = timer()
    d_r.copy_to_host(r)
    d_v.copy_to_host(v)
    d_a.copy_to_host(a)
    d_m.copy_to_host(m)
    dtoh_tf = timer()
    K = 0.0; U = 0.0
    if flag_Energy:
        for i in prange(nobj):
            # Sum the total kinectic energy
            K += 0.5*m[i]*(v[i][0]**2 + v[i][1]**2 + v[i][2]**2)
            for j in range(nobj):
                if (i < j):
                    # Sum the total potential energy
                    U += (-G*m[i]*m[j])/math.sqrt((r[j][0] - r[i][0])**2 + 
                                                  (r[j][1] - r[i][1])**2 + 
                                                  (r[j][2] - r[i][2])**2 +
                                                  sft**2)
    print("Copied data from CPU to GPU after {} s".format(htod_tf - htod_ti))
    print("GPU calculations done after {} s".format(dtoh_ti - htod_tf))
    print("Copied data from GPU to CPU after {} s".format(dtoh_tf - dtoh_ti))
    print("Copies CPU<->GPU and GPU calculations after {} s".format(dtoh_tf - 
                                                                    htod_ti))
    return K, U
    Energies_K_U = np.zeros(2)
    if flag_Energy:
        gravitationalNbody_K_U_NumbaCUDA[blockspergrid, 
threadsperblock](r, v, m, nobj, Energies_K_U)
        cuda.synchronize()
    return Energies_K_U[0], Energies_K_U[1]

# Função de controle para uniformização das chamadas das funções de simulação #

# n_cube_edge : número de pontos l de cada aresta de um cubo
# ninter : M, número de interações
# dt : dt, intervalo entre interações
def run_nbody(gravitationalNbodyfunction, n_cube_edge=10, ninter = 5, dt = 0.01, 
              flag_array=True, flag_compile=True, flag_Energy= False, 
              flag_verbose=False):
    nbody_version = "1.6.3"

    nobj = n_cube_edge**3 # define o número de objetos na simulação
    if flag_array:
        # inicializa uma lista de N posições 3D, nulas inicialmente
        positions = np.zeros((nobj, 3))
        # inicializa uma lista de N velocidades 3D, nulas inicialmente
        velocities = np.zeros_like(positions)
        # inicializa uma lista de N acelerações 3D, nulas inicialmente
        accelerations = np.zeros_like(positions)
        # inicializa uma lista de N massas, aqui iguais a 10^6 kg
        masses = 1e9*np.ones(nobj)   
    else:
        # inicializa uma lista de N de objetos para posição e massa, nulas 
        # inicialmente
        positions = [[0., 0., 0.] for i in range(nobj)]
        # inicializa uma lista de N velocidades 3D, nulas inicialmente
        velocities = [[0., 0., 0.] for i in range(nobj)]
        # inicializa uma lista de N acelerações 3D, nulas inicialmente
        accelerations = [[0., 0., 0.] for i in range(nobj)]
        # inicializa uma lista de N massas, aqui iguais a 10^6 kg
        masses = [1e9]*nobj   
    # Cria um cubo de l^3 pontos, onde cada aresta tem l pontos   
    scale = 1.0  # escala do cubo, com comprimento da aresta = scale*n_cube_edge
    n_half_cube_edge = int(n_cube_edge/2)
    for i in range(n_cube_edge):
        for j in range(n_cube_edge):
            for k in range(n_cube_edge):
                positions[k + j*n_cube_edge + i*n_cube_edge*n_cube_edge] = [
                    (i - n_half_cube_edge)*scale, 
                    (j - n_half_cube_edge)*scale,
                    (k - n_half_cube_edge)*scale]                
    if flag_compile:
        # To Numba compile the function 1st time before calling with full data
        positions_backup = positions.copy()
        velocities_backup = velocities.copy()
        accelerations_backup = accelerations.copy()
        print("N-Body (C) Flavio Manoel, v{}".format(nbody_version))
        print("N = {} corpos, dt = {}s, M = {} interações".format(nobj, dt,
                                                                  ninter))
        start = timer()
        Kf, Uf = gravitationalNbodyfunction(positions, velocities,
                                            accelerations, masses, nobj, ninter,
                                            dt, flag_Energy=flag_Energy)
        end = timer()
        if flag_Energy:
            print("Após {} passos, {:.4f}s, Ef = {}, Kf = {}, Uf =
                  {}".format(ninter, ninter*dt, Kf + Uf, Kf, Uf))
        dt_compile_run = end - start
        print("Código executado em {:.6f} s\n".format(dt_compile_run))
        positions = positions_backup.copy()
        velocities = velocities_backup.copy()
        accelerations = accelerations_backup.copy()
    else:
        dt_compile_run = 0.0
        
    print("N-Body (C) Flavio Manoel, v{}".format(nbody_version))
    print("N = {} corpos, dt = {}s, M = {} interações".format(nobj, dt, ninter))
    if flag_verbose:
        print("Posições = {}\nVelocidades = {}\nAcelerações = {}\nMassas = {}
              \n".format(positions, 
              velocities, accelerations, masses))
    start = timer()
    if flag_Energy:
        Ki, Ui = gravitationalNbodyfunction(positions, velocities,
                                            accelerations, masses, nobj, 0, dt, 
                                            flag_Energy=flag_Energy)
        print("Ei = {}, Ki = {}, Ui = {}".format(Ki + Ui, Ki, Ui))
    Kf, Uf = gravitationalNbodyfunction(positions, velocities, accelerations,
                                        masses, nobj, ninter, dt, 
                                        flag_Energy=flag_Energy)
    end = timer()
    if flag_Energy:
        erro_perc = 100.0*(Kf + Uf - (Ki + Ui))/(Ki + Ui)
        print("Após {} passos, {:.4f}s :\nEf = {}, Kf = {}, Uf = {}, 
              erro em E = {}%".format(ninter,
                                      ninter*dt, Kf + Uf, Kf, Uf, erro_perc))
    else:
        erro_perc = 0.0
    dt_run = end - start
    print("Código executado em {:.6f} s".format(dt_run))
    if flag_verbose:
        print("\nPositions = {}\nVelocities = {}\nAccelerations =
              {}".format(positions, velocities, accelerations))
    return dt_compile_run, dt_run, erro_perc

#------ Definições de constantes da Simulação ------#
n_cube_edge=32; ninter = 1; dt = 0.01

#------ Chamadas das funções de teste ------#

## Teste de nbody em Python puro
t_all_py, t_py, erro_py = run_nbody(gravitationalNbody_NumbaCPU.py_func,
                                    n_cube_edge=n_cube_edge, ninter=ninter, 
                                    dt=dt, flag_array=False, flag_compile=False)

## Teste de nbody em Python NumPy
t_all_np, t_np, erro_np = run_nbody(gravitationalNbody_NumPy_NumbaCPU.py_func,
                                    n_cube_edge=n_cube_edge, 
                                    ninter=ninter, dt=dt, flag_compile=False)

## Teste de nbody em Python Numba CPU
t_all_nb, t_nb, erro_nb = run_nbody(gravitationalNbody_NumbaCPU,
                                    n_cube_edge=n_cube_edge, ninter=ninter, 
                                    dt=dt)

## Teste de nbody em NumPy Numba CPU
t_all_np_nb, t_np_nb, erro_np_nb = run_nbody(gravitationalNbody_NumPy_NumbaCPU,
                                             n_cube_edge=n_cube_edge, 
                                             ninter=ninter, dt=dt)

## Teste de nbody em Python Numba CPU //
t_all_nb_parallel,
t_nb_parallel,
erro_nb_parallel = run_nbody(gravitationalNbody_NumbaCPUparallel,
                             n_cube_edge=n_cube_edge, ninter=ninter, dt=dt)

## Teste de nbody em NumPy Numba CPU //
t_all_np_nb_parallel,
t_np_nb_parallel,
erro_np_nb_parallel = run_nbody(gravitationalNbody_NumPy_NumbaCPUparallel,
                                n_cube_edge=n_cube_edge, ninter=ninter, dt=dt)

## Teste de nbody em Python Numba GPU
t_all_gpu,
t_gpu,
erro_gpu = run_nbody(gravitationalNbody_NumbaGPU64, n_cube_edge=n_cube_edge,
                     ninter=ninter, dt=dt)
