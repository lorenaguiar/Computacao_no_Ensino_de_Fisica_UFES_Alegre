@cuda.jit("(float64[:,:], float64[:,:], float64[:,:], float64[:], int32, float64)")
def gravitationalNbody_a_v_ij_NumbaGPU64(r, v, a, m, nobj, dt):
    i = cuda.grid(1)
    if i < nobj:
        a[i][0] = 0.0; a[i][1] = 0.0; a[i][2] = 0.0
        for j in range(nobj):
            if i != j:
...
