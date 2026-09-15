def ilp(nvr, mr, mc, nc, kc, RoB):
    if nvr * mr * kc >= RoB:
        return nvr * mr
    return min(nvr * mr * mc * nc, RoB / kc)
