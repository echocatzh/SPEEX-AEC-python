# The echo canceller is based on the MDF algorithm described in:
# 
# J. S. Soo, K. K. Pang Multidelay block frequency adaptive filter, 
# IEEE Trans. Acoust. Speech Signal Process., Vol. ASSP-38, No. 2, 
# February 1990.
# 
# We use the Alternatively Updated MDF (AUMDF) variant. Robustness to 
# double-talk is achieved using a variable learning rate as described in:
# 
# Valin, J.-M., On Adjusting the Learning Rate in Frequency Domain Echo 
# Cancellation With Double-Talk. IEEE Transactions on Audio,
# Speech and Language Processing, Vol. 15, No. 3, pp. 1030-1034, 2007.
# http://people.xiph.org/~jm/papers/valin_taslp2006.pdf
# 
# There is no explicit double-talk detection, but a continuous variation
# in the learning rate based on residual echo, double-talk and background
# noise.
# 
# Another kludge that seems to work good: when performing the weight
# update, we only move half the way toward the "goal" this seems to
# reduce the effect of quantization noise in the update phase. This
# can be seen as applying a gradient descent on a "soft constraint"
# instead of having a hard constraint.
# 
# Notes for this file:
#
# Usage: 
#
#    processor = MDF(Fs, frame_size, filter_length)
#    processor.main_loop(u, d)
#    
#    Fs                  sample rate
#    u                   speaker signal, vector in range [-1; 1]
#    d                   microphone signal, vector in range [-1; 1]
#    filter_length       typically 250ms, i.e. 4096 @ 16k FS
#                        must be a power of 2
#    frame_size          typically 8ms, i.e. 128 @ 16k Fs
#                        must be a power of 2
#
# Shimin Zhang <shmzhang@npu-aslp.org>
# 

import numpy as np

def float_to_short(x):
    x = x*32768.0
    x[x < -32767.5] = -32768
    x[x > 32766.5] = 32767
    x = np.floor(0.5+x)
    return x

class MDF:
    def __init__(self, fs: int, frame_size: int, filter_length: int) -> None:
        nb_mic = 1
        nb_speakers = 1
        self.K = nb_speakers
        K = self.K
        self.C = nb_mic
        C = self.C

        self.frame_size = frame_size
        self.filter_length = filter_length
        self.window_size = frame_size*2
        N = self.window_size
        self.M = int(np.fix((filter_length+frame_size-1)/frame_size))
        M = self.M
        self.cancel_count = 0
        self.sum_adapt = 0
        self.saturated = 0
        self.screwed_up = 0

        self.sampling_rate = fs
        self.spec_average = (self.frame_size)/(self.sampling_rate)
        self.beta0 = (2.0*self.frame_size)/self.sampling_rate
        self.beta_max = (.5*self.frame_size)/self.sampling_rate
        self.leak_estimate = 0

        self.e = np.zeros((N, C),)
        self.x = np.zeros((N, K),)
        self.input = np.zeros((self.frame_size, C),)
        self.y = np.zeros((N, C),)
        self.last_y = np.zeros((N, C),)
        self.Yf = np.zeros((self.frame_size+1, 1),)
        self.Rf = np.zeros((self.frame_size+1, 1),)
        self.Xf = np.zeros((self.frame_size+1, 1),)
        self.Yh = np.zeros((self.frame_size+1, 1),)
        self.Eh = np.zeros((self.frame_size+1, 1),)

        self.X = np.zeros((N, K, M+1), dtype=np.complex)
        self.Y = np.zeros((N, C), dtype=np.complex)
        self.E = np.zeros((N, C), dtype=np.complex)
        self.W = np.zeros((N, K, M, C), dtype=np.complex)
        self.foreground = np.zeros((N, K, M, C), dtype=np.complex)
        self.PHI = np.zeros((frame_size+1, 1),)
        self.power = np.zeros((frame_size+1, 1),)
        self.power_1 = np.ones((frame_size+1, 1),)
        self.window = np.zeros((N, 1),)
        self.prop = np.zeros((M, 1),)
        self.wtmp = np.zeros((N, 1),)
        self.window = .5-.5 * \
            np.cos(2*np.pi*(np.arange(1, N+1).reshape(-1, 1)-1)/N)
        decay = np.exp(-2.4/M)
        self.prop[0, 0] = .7
        for i in range(1, M):
            self.prop[i, 0] = self.prop[i-1, 0]*decay
        self.prop = (.8 * self.prop)/np.sum(self.prop)

        self.memX = np.zeros((K, 1),)
        self.memD = np.zeros((C, 1),)
        self.memE = np.zeros((C, 1),)
        self.preemph = .9
        if self.sampling_rate < 12000:
            self.notch_radius = .9
        elif self.sampling_rate < 24000:
            self.notch_radius = .982
        else:
            self.notch_radius = .992
        self.notch_mem = np.zeros((2*C, 1),)
        self.adapted = 0
        self.Pey = 1
        self.Pyy = 1
        self.Davg1 = 0
        self.Davg2 = 0
        self.Dvar1 = 0
        self.Dvar2 = 0

    def main_loop(self, u, d):
        """MDF core function

        Args:
            u (array): reference signal
            d (array): microphone signal
        """
        assert u.shape == d.shape
        u = float_to_short(u)
        d = float_to_short(d)

        e = np.zeros_like(u)
        y = np.zeros_like(u)
        end_point = len(u)

        for n in range(0, end_point, self.frame_size):
            nStep = np.floor(n/self.frame_size) + 1
            self.nStep = nStep
            # the break operation not understand.
            # only for signal channel AEC
            if n+self.frame_size > end_point:
                break
            u_frame = u[n:n+self.frame_size]
            d_frame = d[n:n+self.frame_size]
            out = self.speex_echo_cancellation_mdf(d_frame[:, None], u_frame[:, None])[:,0]
            e[n:n+self.frame_size] = out
            y[n:n+self.frame_size] = d_frame - out
        e = e/32768.0
        y = y/32768.0
        return e, y

    def speex_echo_cancellation_mdf(self, mic, far_end):
        N = self.window_size
        M = self.M
        C = self.C
        K = self.K

        Pey_cur = 1
        Pyy_cur = 1

        out = np.zeros((self.frame_size, C),)
        self.cancel_count += 1

        ss = .35/M
        ss_1 = 1 - ss

        for chan in range(C):
            # Apply a notch filter to make sure DC doesn't end up causing problems
            self.input[:, chan], self.notch_mem[:, chan] = self.filter_dc_notch16(
                mic[:, chan], self.notch_mem[:, chan])

            for i in range(self.frame_size):
                tmp32 = self.input[i, chan] - \
                    (np.dot(self.preemph, self.memD[chan]))
                self.memD[chan] = self.input[i, chan]
                self.input[i, chan] = tmp32

        for speak in range(K):
            for i in range(self.frame_size):
                self.x[i, speak] = self.x[i+self.frame_size, speak]
                tmp32 = far_end[i, speak] - \
                    np.dot(self.preemph, self.memX[speak])
                self.x[i+self.frame_size, speak] = tmp32
                self.memX[speak] = far_end[i, speak]

        # self.X = np.roll(self.X, [0, 0, 1])
        self.X = np.roll(self.X, 1, axis=2)

        for speak in range(K):
            self.X[:, speak, 0] = np.fft.fft(self.x[:, speak])/N
        
        Sxx = 0
        for speak in range(K):
            Sxx = Sxx + np.sum(self.x[self.frame_size:, speak]**2)
            self.Xf = np.abs(self.X[:self.frame_size+1, speak, 0])**2
        Sff = 0
        for chan in range(C):
            self.Y[:, chan] = 0
            for speak in range(K):
                for j in range(M):
                    self.Y[:, chan] = self.Y[:, chan] + self.X[:,
                                                               speak, j]*self.foreground[:, speak, j, chan]
            self.e[:, chan] = np.fft.ifft(self.Y[:, chan]).real * N
            self.e[:self.frame_size, chan] = self.input[:, chan] - \
                self.e[self.frame_size:, chan]
            Sff = Sff + np.sum(np.abs(self.e[:self.frame_size, chan])**2)

        if self.adapted:
            self.prop = self.mdf_adjust_prop()
        if self.saturated == 0:
            for chan in range(C):
                for speak in range(K):
                    for j in list(range(M)[::-1]):
                        self.PHI = np.concatenate([self.power_1, self.power_1[-2:0:-1]], axis=0) * self.prop[j] * np.conj(self.X[:, speak, j+1])[:,None] * self.E[:, chan][:,None]
                        self.W[:,speak,j,chan] = self.W[:,speak,j,chan]+self.PHI[:,0]

        else:
            self.saturated -= 1

        for chan in range(C):
            for speak in range(K):
                for j in range(M):
                    if j == 0 or (2+self.cancel_count) % (M-1) == j:
                        self.wtmp = np.fft.ifft(self.W[:, speak, j, chan]).real
                        self.wtmp[self.frame_size:N] = 0
                        self.W[:, speak, j, chan] = np.fft.fft(self.wtmp)

        self.Yf = np.zeros((self.frame_size+1, 1),)
        self.Rf = np.zeros((self.frame_size+1, 1),)
        self.Xf = np.zeros((self.frame_size+1, 1),)

        Dbf = 0
        for chan in range(C):
            self.Y[:, chan] = 0
            for speak in range(K):
                for j in range(M):
                    self.Y[:, chan] = self.Y[:, chan] + \
                        self.X[:, speak, j] * self.W[:, speak, j, chan]
            self.y[:, chan] = np.fft.ifft(self.Y[:, chan]).real*N

        See = 0

        for chan in range(C):
            self.e[:self.frame_size, chan] = self.e[self.frame_size:N,
                                                    chan]-self.y[self.frame_size:N, chan]

                
            Dbf = Dbf + 10 + np.sum(np.abs(self.e[:self.frame_size, chan])**2)
            self.e[:self.frame_size, chan] = self.input[:, chan] - \
                self.y[self.frame_size:N, chan]
            See = See + np.sum(np.abs(self.e[:self.frame_size, chan])**2)
        
        VAR1_UPDATE = .5
        VAR2_UPDATE = .25
        VAR_BACKTRACK = 4
        MIN_LEAK = .005

        self.Davg1 = .6*self.Davg1 + .4*(Sff-See)
        self.Dvar1 = .36*self.Dvar1 + .16*Sff*Dbf
        self.Davg2 = .85*self.Davg2 + .15*(Sff-See)
        self.Dvar2 = .7225*self.Dvar2 + .0225*Sff*Dbf

        update_foreground = 0
        if (Sff-See)*abs(Sff-See) > (Sff*Dbf):
            update_foreground = 1
        elif (self.Davg1 * abs(self.Davg1) > (VAR1_UPDATE*self.Dvar1)):
            update_foreground = 1
        elif (self.Davg2 * abs(self.Davg2) > (VAR2_UPDATE*(self.Dvar2))):
            update_foreground = 1

        if update_foreground:
            self.Davg1 = 0
            self.Davg2 = 0
            self.Dvar1 = 0
            self.Dvar2 = 0
            self.foreground = self.W
            for chan in range(C):
                self.e[self.frame_size:N, chan] = (self.window[self.frame_size:N][:,0] * self.e[self.frame_size:N, chan]) + (
                    self.window[:self.frame_size][:,0] * self.y[self.frame_size:N, chan])
        else:
            reset_background = 0
            if (-(Sff-See)*np.abs(Sff-See) > VAR_BACKTRACK*(Sff*Dbf)):
                reset_background = 1
            if ((-self.Davg1 * np.abs(self.Davg1)) > (VAR_BACKTRACK*self.Dvar1)):
                reset_background = 1
            if ((-self.Davg2 * np.abs(self.Davg2)) > (VAR_BACKTRACK*self.Dvar2)):
                reset_background = 1

            if reset_background:
                self.W = self.foreground
                for chan in range(C):

                    self.y[self.frame_size:N,
                           chan] = self.e[self.frame_size:N, chan]
                    self.e[:self.frame_size, chan] = self.input[:,
                                                                chan] - self.y[self.frame_size:N, chan]
                See = Sff
                self.Davg1 = 0
                self.Davg2 = 0
                self.Dvar1 = 0
                self.Dvar2 = 0

        Sey = 0
        Syy = 0
        Sdd = 0

        for chan in range(C):
            for i in range(self.frame_size):
                tmp_out = self.input[i, chan] - self.e[i+self.frame_size, chan]
                tmp_out = tmp_out + self.preemph * self.memE[chan]
                # This is an arbitrary test for saturation in the microphone signal
                if mic[i, chan] <= -32000 or mic[i, chan] >= 32000:
                    if self.saturated == 0:
                        self.saturated = 1
                out[i, chan] = tmp_out[0]
                self.memE[chan] = tmp_out

            self.e[self.frame_size:N, chan] = self.e[:self.frame_size, chan]
            self.e[:self.frame_size, chan] = 0
            Sey = Sey + np.sum(self.e[self.frame_size:N, chan]
                               * self.y[self.frame_size:N, chan])
            Syy = Syy + np.sum(self.y[self.frame_size:N, chan]**2)
            Sdd = Sdd + np.sum(self.input**2)

            self.E = np.fft.fft(self.e,axis=0) / N

            self.y[:self.frame_size, chan] = 0
            self.Y = np.fft.fft(self.y,axis=0) / N
            self.Rf = np.abs(self.E[:self.frame_size+1, chan])**2
            self.Yf = np.abs(self.Y[:self.frame_size+1, chan])**2
        if not (Syy >= 0 and Sxx >= 0 and See >= 0):
            self.screwed_up = self.screwed_up + 50
            out = np.zeros_like(out)
        elif Sff > Sdd + N * 10000:
            self.screwed_up = self.screwed_up + 1
        else:
            self.screwed_up = 0
        if self.screwed_up >= 50:
            print("Screwed up, full reset")
            self.__init__(self.sampling_rate,
                          self.frame_size, self.filter_length)

        See = max(See, N * 100)
        for speak in range(K):
            Sxx = Sxx + np.sum(self.x[self.frame_size:, speak]**2)
            self.Xf = np.abs(self.X[:self.frame_size+1, speak, 0])**2
        self.power = ss_1*self.power + 1 + ss*self.Xf[:,None]
        Eh_cur = self.Rf - self.Eh
        Yh_cur = self.Yf - self.Yh
        Pey_cur = Pey_cur + np.sum(Eh_cur*Yh_cur)
        Pyy_cur = Pyy_cur + np.sum(Yh_cur**2)
        self.Eh = (1-self.spec_average)*self.Eh + self.spec_average*self.Rf
        self.Yh = (1-self.spec_average)*self.Yh + self.spec_average*self.Yf
        Pyy = np.sqrt(Pyy_cur)
        Pey = Pey_cur/Pyy
        tmp32 = self.beta0*Syy
        if tmp32 > self.beta_max*See:
            tmp32 = self.beta_max*See
        alpha = tmp32 / See
        alpha_1 = 1 - alpha
        self.Pey = alpha_1*self.Pey + alpha*Pey
        self.Pyy = alpha_1*self.Pyy + alpha*Pyy
        if self.Pyy < 1:
            self.Pyy = 1
        if self.Pey < MIN_LEAK * self.Pyy:
            self.Pey = MIN_LEAK * self.Pyy
        if self.Pey > self.Pyy:
            self.Pey = self.Pyy
        self.leak_estimate = self.Pey/self.Pyy
        if self.leak_estimate > 16383:
            self.leak_estimate = 32767
        RER = (.0001*Sxx + 3.*self.leak_estimate*Syy) / See
        if RER < Sey*Sey/(1+See*Syy):
            RER = Sey*Sey/(1+See*Syy)
        if RER > .5:
            RER = .5
        if (not self.adapted and self.sum_adapt > M and self.leak_estimate*Syy > .03*Syy):
            self.adapted = 1

        if self.adapted:
            for i in range(self.frame_size+1):
                r = self.leak_estimate*self.Yf[i]
                e = self.Rf[i]+1
                if r > .5*e:
                    r = .5*e
                r = 0.7*r + 0.3*(RER*e)
                self.power_1[i] = (r/(e*self.power[i]+10))
        else:
            adapt_rate = 0
            if Sxx > N * 1000:
                tmp32 = 0.25 * Sxx
                if tmp32 > .25*See:
                    tmp32 = .25*See
                adapt_rate = tmp32 / See
            self.power_1 = adapt_rate/(self.power+10)
            self.sum_adapt = self.sum_adapt+adapt_rate

        self.last_y[:self.frame_size] = self.last_y[self.frame_size:N]
        if self.adapted:
            self.last_y[self.frame_size:N] = mic-out
        return out

    def filter_dc_notch16(self, mic, mem):
        out = np.zeros_like(mic)
        den2 = self.notch_radius**2 + 0.7 * \
            (1-self.notch_radius)*(1 - self.notch_radius)
        for i in range(self.frame_size):
            vin = mic[i]
            vout = mem[0] + vin
            mem[0] = mem[1] + 2*(-vin + self.notch_radius*vout)
            mem[1] = vin - (den2*vout)
            out[i] = self.notch_radius * vout
        return out, mem

    def mdf_adjust_prop(self,):
        N = self.window_size
        M = self.M
        C = self.C
        K = self.K
        prop = np.zeros((M, 1),)
        for i in range(M):
            tmp = 1
            for chan in range(C):
                for speak in range(K):
                    tmp = tmp + np.sum(np.abs(self.W[:N//2+1, speak, i, chan])**2)
            prop[i] = np.sqrt(tmp)
        max_sum = np.maximum(prop, 1)
        prop = prop + .1 * max_sum
        prop_sum = 1 + np.sum(prop)
        prop = 0.99*prop/prop_sum
        return prop


if __name__ == "__main__":
    import soundfile as sf
    import librosa
    mic, sr = sf.read("samples/mic.wav")
    ref, sr = sf.read("samples/lpb.wav")
    min_len = min(len(mic), len(ref))
    mic = mic[:min_len]
    ref = ref[:min_len]
    # 64 2048 for 8kHz.
    processor = MDF(sr, 128, 4096)
    e, y = processor.main_loop(ref, mic)
    sf.write('e.wav', e, sr)
    sf.write('y.wav', y, sr)
