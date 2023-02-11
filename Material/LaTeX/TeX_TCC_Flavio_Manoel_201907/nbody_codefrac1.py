def gravitationalNbody(r, v, a, m, nobj, ninter, dt):
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
