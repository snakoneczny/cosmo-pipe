import numpy as np
from scipy.special import erf


class FluxPDF(object):
    def __init__(self, fname_in="data/skads_flux_counts.result"):
        from scipy.interpolate import interp1d
        # Read flux distribution from SKADS' S3-SEX simulation
        self.log_flux, counts = np.loadtxt(fname_in, unpack=True,
                                           delimiter=',', skiprows=1)
        self.log_flux += 3  # Use mJy instead of Jy
        # Assuming equal spacing
        self.dlog_flux = np.mean(np.diff(self.log_flux))
        self.log_flux = self.log_flux[counts >= 0]
        counts = counts[counts >= 0]
        self.probs = counts / np.sum(counts)
        # Cut to non-zero counts
        self.lpdf = interp1d(self.log_flux, np.log10(counts),
                             fill_value=-500, bounds_error=False)

    def plot_pdf(self, log_flux_min=-6, log_flux_max=6,
                 n_log_flux=256):
        lf = np.linspace(log_flux_min, log_flux_max, n_log_flux)
        dlf = np.mean(np.diff(lf))
        p = 10.**self.lpdf(lf)
        p /= np.sum(p) * dlf
        plt.figure()
        plt.plot(10.**lf, p, 'k-')
        plt.loglog()
        plt.xlabel(r'$I_{1400}\,{\rm mJy}$', fontsize=14)
        plt.ylabel(r'$dp/d\log_{10}I_{1400}$', fontsize=14)
        plt.show()

    def compute_p_values(self, q, std_map, Imin, alpha=-0.7):
        lf = self.log_flux + alpha * np.log10(144. / 1400.)
        p_map = np.zeros(len(std_map))
        for ip, std in enumerate(std_map):
            if std > 0:
                Ithr = max(q * std, Imin)
                x = (Ithr - 10.**lf) / (np.sqrt(2.) * std)
                comp = 0.5 * (1 - erf(x))
                p_map[ip] = np.sum(self.probs * comp)
        return p_map

    def draw_random_fluxes(self, n, alpha=-0.7, lf_thr_low=-3.5):
        msk = self.log_flux >= lf_thr_low
        lf_ax = self.log_flux[msk]
        p_ax = self.probs[msk]
        p_ax /= np.sum(p_ax)
        lf = np.random.choice(lf_ax, size=n, p=p_ax)
        lf += self.dlog_flux * (np.random.random(n)-0.5)
        # Extrapolate to 144 MHz
        # Assumption: I_nu = I_1400 * (nu / 1400)^alpha
        if alpha != 0:
            lf += alpha * np.log10(144. / 1400.)
        return lf
