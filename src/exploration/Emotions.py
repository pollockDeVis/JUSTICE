import numpy as np


class Emotion_opinion:
    def __init__(self, valence, opinion):
        """
        region_code::string::code of the region in JUSTICE. Ex. 'rcam'
        valence::float::valence of emotion regarding the specific issue
        opinion::float::opinion regarding the specific issue
        """
        # Emotions dynamics (valence, arousal, sharing)
        self.v = valence; #Random init: 0.3 * np.random.normal(0, 0.5, 1)[0]
        self.a = 0
        self.s = 0
        self.dt = 0.2  # !!! dt is also defined again in OPINIONS.py (ensure it is the same value)
        # Opinions dynamics (opinion)
        self.o = opinion
        self.count_o_opposed_v = 0

        # Emotion parameters
        # B for valence
        self.b0 = 0  # always equals 0 but can be use to bias the emotion
        self.b1 = 1
        # b2 = 0 (other values can be used to add an emotional bias, does not work very well --> use b0 instead)
        self.b2 = 0
        self.b3 = -1
        # D for arousal
        # Small arousal bias, to ensure communication of emotions
        self.d0 = 0.1
        self.d1 = 0.5
        self.d2 = 0.1
        self.gamma_a = 0.9
        self.gamma_v = 0
        self.randcoeff_a = 0.3
        self.randcoeff_v = 0.3
        # tau min = 0.1 and tau max = 1.1
        # self.tau = np.random.random() + 0.1
        self.tau = 0.6

        # Opinion parameters
        self.c0 = 0.1
        self.c1 = 1
        # self.alpha0 = 0 using an explicit expression depending on emotions
        # self.alpha1 = 0 using an explicit expression depending on emotions
        # alpha2 = 0, other values can be used to induce a bias in the opinion
        self.alpha2 = 0
        self.alpha3 = -3
        self.randcoeff_o = 0.05
        self.h_base = 0.1

    def step(self, h, h_plus, h_minus, v_mean):
        # On peut faire lentement progresser alpha2 au cours de la simulation
        self.step_opinion(h, v_mean)
        self.step_emotion(h, h_plus, h_minus)

    def step_emotion(self, h, h_plus, h_minus):
        self.update_a(h)
        self.update_v(h_plus, h_minus)
        self.update_s()

    def step_opinion(self, h, v_mean):
        self.update_o(h, v_mean)

    def g_valence(self, h_plus, h_minus):
        # b0 decay
        self.b0 = 0.95 * self.b0
        if self.v >= 0:
            return (
                1
                / 3
                * (h_plus)
                * (
                    self.b1 * self.v
                    + self.b3 * self.v**3
                    + self.b2 * self.v**2
                    + self.b0
                )
            )
        else:
            return (
                1
                / 3
                * (-h_minus)
                * (
                    self.b1 * self.v
                    + self.b3 * self.v**3
                    + self.b2 * self.v**2
                    - self.b0
                )
            )

    def g_arousal(self, h):
        return h * (self.d0 + self.d1 * self.a + self.d2 * self.a**2)

    def update_a(self, h):
        if self.tau - self.a >= 0:
            self.a = self.a + self.dt * (
                -self.gamma_a * self.a
                + self.g_arousal(h)
                + self.randcoeff_a * np.random.normal(0, 6, 1)[0]
            )
        else:
            self.a = 0
            self.alpha3 = -3

    def update_v(self, h_plus, h_minus):
        self.v = self.v + self.dt * (
            -self.gamma_v * self.v
            + self.g_valence(h_plus, h_minus)
            # + self.randcoeff_v * np.random.normal(0, 0.5, 1)[0]
        )

    def update_s(self):
        self.s = np.sign(self.v) * np.heaviside(self.a - self.tau, 1)

    def update_o(self, h, v_mean):
        if self.o * self.v < 0:
            self.count_o_opposed_v = self.count_o_opposed_v + 1
            self.o = self.o * (1 - np.abs(self.v)*0.01)
            self.alpha3 = self.alpha3 * (1 + self.count_o_opposed_v * np.abs(self.v)*0.01)
        else:
            self.count_o_opposed_v = 0
            self.alpha3 = -3

        self.o = np.clip(
            self.o
            + self.dt
            * (
                -self.c0**2 * h * -1 * (v_mean + self.v) / 2
                + self.c1 * (h - self.h_base) * self.o
                + self.alpha2 * self.o**2
                + self.alpha3 * self.o**3
                # + self.randcoeff_o * np.random.normal(0, 0.3, 1)[0]
            ),
            -2,
            2,
        )
