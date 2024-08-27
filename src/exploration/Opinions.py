import numpy as np
from src.exploration.Emotions import Emotion_opinion
class Opinion:
    def __init__(self, n_agents, agents):
        self.n_agents = n_agents
        self.dt = 0.2 #!!! dt is also defined again in EMOTIONS.py (ensure it is the same value)
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
        # Not used in Schweitzer, Frank; Krivachy, Tamas; Garcia, David 2020, not used here either
        self.I_minus = 0
        self.I_plus = 0
        # Decay of field
        self.gamma_plus = 0.7
        self.gamma_minus = 0.7

        self.agents = agents

    def update_h_minus(self):
        self.h_minus = self.h_minus + self.dt * (
            -self.gamma_minus * self.h_minus + self.s * self.N_minus + self.I_minus
        )

    def update_h_plus(self):
        self.h_plus = self.h_plus + self.dt * (
            -self.gamma_plus * self.h_plus + self.s * self.N_plus + self.I_plus
        )

    def update_h(self):
        self.h = self.h_minus + self.h_plus

    def compute_delta_h(self):
        return self.h_plus - self.h_minus

    def update_N(self, agents):
        self.N_minus = sum([a.s < 0 for a in self.agents])*100/self.n_agents
        self.N_plus = sum([a.s > 0 for a in self.agents])*100/self.n_agents

    def compute_v_mean(self, agents):
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

    def compute_a_mean(self, agents):
        a_mean = 0
        for agent in self.agents:
            a_mean = a_mean + agent.a
        return a_mean / self.n_agents

    def step(self):
        # Update the internal states of agents
        count_Sharing_positive = 0
        count_Sharing_negative = 0
        for a in self.agents:
            #v_mean = self.compute_v_mean(self.agents)
            a.step(
                self.h,
                self.h_plus,
                self.h_minus,
                self.compute_delta_h(),
            )
            #counting positive and negative sharings here, to avoid looping over the agents twice
            count_Sharing_positive += a.s>0
            count_Sharing_negative += a.s<0
        #Updating the N
        self.N_minus = count_Sharing_negative
        self.N_plus = count_Sharing_positive
        #self.update_N(agents)
        self.update_h_minus()
        self.update_h_plus()
        self.update_h()