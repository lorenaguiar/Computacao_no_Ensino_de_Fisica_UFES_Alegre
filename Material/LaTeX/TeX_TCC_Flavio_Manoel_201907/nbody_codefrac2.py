...
    K = 0.0; U = 0.0
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
