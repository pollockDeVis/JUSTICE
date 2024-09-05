import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dt = 0.2


# The parameters for bias are:
# b2 for the emotional bias
# alpha2 for the opinion bias
# As dt = 0.2, you have to run the simulation for 5 steps each years
class EmotionOpinions:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        # emotion field
        self.h_minus = 0
        self.h_plus = 0
        self.h = 0
        # Number of agents sharing emotions
        self.N_minus = 0
        self.N_plus = 0
        # Social impact factor (identical for each agent, but could vary by agent, then h_minus and h_plus become vectors)
        # value is 0.1 (2010) or 0.6 (2020)
        self.s = 0.1
        # Influence from news and medias (identical for each agent, but could vary by agent, then h_minus and h_plus become vectors)
        self.I_minus = 0
        self.I_plus = 0
        # Decay of field
        self.gamma_plus = 0.7
        self.gamma_minus = 0.7

        # test
        self.decay_arousal = 1

        # Agents
        self.agents = [Agents() for i in range(n_agents)]

    def update_h_minus(self):
        self.h_minus = self.h_minus + dt * (
            -self.gamma_minus * self.h_minus + self.s * self.N_minus + self.I_minus
        )

    def update_h_plus(self):
        self.h_plus = self.h_plus + dt * (
            -self.gamma_plus * self.h_plus + self.s * self.N_plus + self.I_plus
        )

    def update_h(self):
        self.h = self.h_minus + self.h_plus

    def compute_delta_h(self):
        return self.h_plus - self.h_minus

    def update_N(self):
        self.N_minus = sum([a.s < 0 for a in self.agents])
        self.N_plus = sum([a.s > 0 for a in self.agents])
        # print(self.N_minus, self.N_plus)

    def compute_v_mean(self):
        v_mean = 0
        v_mean_plus = 0
        v_mean_minus = 0
        count = 0
        for agent in self.agents:
            if agent.a > agent.tau:
                v_mean = v_mean + agent.v
                if agent.v > 0:
                    v_mean_plus = v_mean_plus + agent.v
                elif agent.v < 0:
                    v_mean_minus = v_mean_minus + agent.v
                count = count + 1
        return [
            v_mean / (count + 0.01),
            v_mean_plus / (self.N_plus + 0.01),
            v_mean_minus / (0.01 + self.N_minus),
        ]

    def compute_a_mean(self):
        a_mean = 0
        for agent in self.agents:
            a_mean = a_mean + agent.a
        return a_mean / self.n_agents

    def step(self):
        # Update the internal states of agents
        for a in self.agents:
            # a.step(self.h, self.h_plus, self.h_minus, self.compute_v_mean())
            v_mean = self.compute_v_mean()
            a.step(
                self.h,
                self.h_plus,
                self.h_minus,
                v_mean[1],
                v_mean[2],
                self.compute_delta_h(),
                self.decay_arousal,
            )
        self.update_N()
        self.update_h_minus()
        self.update_h_plus()
        self.update_h()


class Agents:
    def __init__(self):
        # Emotions dynamics (valence, arousal, sharing)
        self.v = 0.3 * np.random.normal(0, 0.5, 1)[0]
        self.a = 0
        self.s = 0

        # Opinions dynamics (opinion)
        self.o = 0
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

    def step(
        self, h, h_plus, h_minus, v_mean_plus, v_mean_minus, v_mean, decay_arousal
    ):
        # On peut faire lentement progresser alpha2 au cours de la simulation
        self.step_opinion(h, h_plus, h_minus, v_mean_plus, v_mean_minus, v_mean)
        self.step_emotion(h, h_plus, h_minus, decay_arousal)

    def step_emotion(self, h, h_plus, h_minus, decay_arousal):
        self.update_a(h, decay_arousal)
        self.update_v(h_plus, h_minus)
        self.update_s()

    def step_opinion(self, h, h_plus, h_minus, v_mean_plus, v_mean_minus, v_mean):
        self.o = self.v
        # self.update_o(h, h_plus, h_minus, v_mean_plus, v_mean_minus, v_mean)

    def g_valence(self, h_plus, h_minus):
        # b0 decay
        self.b0 = 0.975 * self.b0
        self.b2 = 0.99 * self.b2
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
            ) + (
                1
                / 6
                * (-h_minus)
                * (self.b1 * self.v + self.b3 * self.v**3 + self.b2 * self.v**2)
            )
        else:
            return (
                1
                / 3
                * (h_minus)
                * (
                    self.b1 * self.v
                    + self.b3 * self.v**3
                    + self.b2 * self.v**2
                    + self.b0
                )
            ) + (
                1
                / 6
                * (-h_plus)
                * (self.b1 * self.v + self.b3 * self.v**3 + self.b2 * self.v**2)
            )

    def g_arousal(self, h):
        return h * (self.d0 + self.d1 * self.a + self.d2 * self.a**2)

    def update_a(self, h, decay_arousal):
        if self.tau - self.a >= 0:
            self.a = self.a + dt * (
                -self.gamma_a * self.a
                + self.g_arousal(h)
                + self.randcoeff_a * np.random.normal(0, 6)
            )
            self.a = self.a * decay_arousal
        else:
            self.a = 0
            self.alpha3 = -3

    def update_v(self, h_plus, h_minus):
        self.v = self.v + dt * (
            -self.gamma_v * self.v
            + self.g_valence(h_plus, h_minus)
            + 0*np.random.normal(0, 0.01)
        )

    def update_s(self):
        self.s = np.sign(self.v) * np.heaviside(self.a - self.tau, 1)

    def update_o(self, h, h_plus, h_minus, v_mean_plus, v_mean_minus, v_mean):
        if self.o * self.v < 0:
            self.count_o_opposed_v = self.count_o_opposed_v + 1
            self.o = self.o * (1 - 0.01)
            self.alpha3 = self.alpha3 * (1 + self.count_o_opposed_v * 0.01)
        else:
            self.count_o_opposed_v = 0
            self.alpha3 = -3

        self.o = np.clip(
            self.o
            + dt
            * (
                -self.c0**2 * h * -1 * (v_mean + self.v) / 2
                + self.c1 * (h - self.h_base) * self.o
                + self.alpha2 * self.o**2
                + self.alpha3 * self.o**3
                # + self.randcoeff_o * np.random.normal(0, 0.3, 1)[0]
            ),
            -1,
            1,
        )


#################################################################################################
#################################################################################################
#################################      SCRIPT      ##############################################
#################################################################################################
#################################################################################################
n_agents = 100
n_iter = 500
time_space = np.arange(0, n_iter, 1)

# Data structure for visualization
df_opinion = pd.DataFrame(columns=[i for i in range(100)])
df_arousal = pd.DataFrame(columns=[i for i in range(n_agents)])
df_valence = pd.DataFrame(columns=[i for i in range(n_agents)])
df_singular = pd.DataFrame(columns=["h", "delta_h", "a_mean", "N_plus", "N_minus"])

if 0:
    for n in range(25):
        print(n)
        model = EmotionOpinions(n_agents)
        for i in range(n_iter):
            model.step()
        opinion_vector = [a.o for a in model.agents]
        valence_vector = [a.v for a in model.agents]
        df_opinion[n] = opinion_vector
        df_valence[n] = valence_vector
    description = df_valence.describe(include="all")
    print("-----------==== VALENCE ====-----------")
    # print(description)
    print(description.mean(axis=1))
    description = df_opinion.describe(include="all")
    print("-----------==== OPINION ====-----------")
    # print(description)
    print(description.mean(axis=1))
    exit()

# Simulation
model = EmotionOpinions(n_agents)
for i in range(n_iter):

    # model.decay_arousal = np.abs(np.sin(i/10.))

    if (i + 1) % 100 == 0:
        for a in model.agents:
            # a.alpha2 = a.alpha2 + 1
            if np.random.random() < 0.5:
                # a.alpha2 = 1  # affects opinion directly (biased the whole opinion spectrum up or down0
                a.b0 += 1  # affects the valence directly
                # a.alpha3 = min(-3, 3 * (a.o - 1))  # affects the nature of the polarization
            if np.random.random() > 0.5:
                # a.alpha2 = 1  # affects opinion directly (biased the whole opinion spectrum up or down0
                a.b0 += 0  # affects the valence directly
                # a.alpha3 = min(-3, 3 * (a.o - 1))  # affects the nature of the polarization

    """elif (i) % 100 == 0:
        for a in model.agents:
            a.alpha2 = 0  # affects opinion directly (biased the whole opinion spectrum up or down0
            a.b0 = 0  # affects the valence directly
            #a.alpha3 = min(-3, 3 * (a.o - 1))  # affects the nature of the polarization"""

    model.step()

    opinion_vector = [a.o for a in model.agents]
    arousal_vector = [a.a for a in model.agents]
    valence_vector = [a.v for a in model.agents]
    df_opinion.loc[len(df_opinion)] = opinion_vector
    # df_opinion.index = df_opinion.index + 1
    df_arousal.loc[len(df_arousal)] = arousal_vector
    df_valence.loc[len(df_valence)] = valence_vector
    # df_arousal.index = df_arousal.index + 1
    df_singular.loc[len(df_singular)] = [
        model.h,
        model.compute_delta_h(),
        model.compute_a_mean(),
        model.N_plus,
        model.N_minus,
    ]
    # df_singular.index = df_singular.index + 1
    print(i)

# Visualization
plt.figure()
print(df_opinion.head())
df_opinion.plot(title="Opinion")
ax = plt.gca()
# ax.set_ylim([-1, 1])
plt.figure()
df_arousal[[10, 20, 30, 40, 50]].plot(title="Arousal")
ax = plt.gca()
# ax.set_ylim([-1, 1])
plt.figure()
df_valence.plot(title="Valence")
ax = plt.gca()
# ax.set_ylim([-1, 1])
plt.figure()
df_singular[["h", "delta_h", "a_mean"]].plot()
ax = plt.gca()
plt.figure()
df_singular[["N_plus", "N_minus"]].plot()
ax = plt.gca()
# ax.set_ylim([-1, 1])
plt.figure()
df_opinion.iloc[-1].plot.hist(bins=12, alpha=0.5, title="Final Opinion")
plt.figure()
df_opinion.iloc[99].plot.hist(bins=12, alpha=0.5, title="Opinion at 99")
plt.figure()
df_valence.iloc[-1].plot.hist(bins=12, alpha=0.5, title="Final Valence")
plt.show()
