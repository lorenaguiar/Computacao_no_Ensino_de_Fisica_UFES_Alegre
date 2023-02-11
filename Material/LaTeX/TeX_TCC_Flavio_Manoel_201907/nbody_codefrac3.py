@njit(parallel=True)
def gravitationalNbody(r, v, a, m, nobj, ninter, dt):
    for inter in range(ninter):
...
