import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
from mesa.batchrunner import BatchRunner
from scipy.stats import weibull_min, dirichlet
import random
import operator
import copy
from scipy.stats import truncnorm


# Basic agents and model

class Patient(Agent):
    """An agent with prob of death."""
    def __init__(self, unique_id, model, doctor=None, stake=None):
        super().__init__(unique_id, model)
        # Weibull params
        self.shape = np.random.uniform(0.5, 3)
        self.scale = 50
        # Intervention effect params
        self.a = np.random.uniform(1.5, 4)
        self.b = 2
        self.time = 0
        self.death = 0 # Probability of death at each timestep
        self.doctor = doctor
        # Truth based on doctor's prior
        self.truth = np.random.choice(self.doctor.hyp, 1, p=self.doctor.cred)
        self.score = np.nan
        self.active = False
        self.sensitivity = np.nan
        self.stake = stake

    def step(self):
        self.time += 1
        self.death = weibull_min(c=self.shape, loc=0, scale=self.scale).cdf(self.time)



class Doctor(Agent):
    """Agent with credence for each hypothesis."""
    def __init__(self, unique_id, model, patient=None, threshold=0.9, N_hyp=11):
        super().__init__(unique_id, model)
        self.N_hyp = N_hyp
        self.dsize = N_hyp # size of domain
        self.active = True
        self.hyp = np.round(np.arange(0, 1.001, 1/(N_hyp-1)), 2) # Array of hypotheses
        alpha = max(np.random.normal(loc=0.5, scale=0.1), 0.1) # param for dirichlet
        self.cred = np.random.dirichlet(np.ones(N_hyp)*alpha,size=1)[0] # Credence for each hypothesis
        self.score = 0
        self.patient = patient
        self.threshold = threshold
        self.sensitivity = None
        self.accuracy = 0
        self.speed = np.nan
        self.pdeath = np.nan # patient's prob of death

    def intervene(self, conclusion=None):
        x = self.patient.stake
        if conclusion is None: # no intervention
            self.score += (1 - self.patient.death)*x
        elif conclusion == self.patient.truth: # correct
            self.score = (1 - (self.patient.death/self.patient.a))*x
            self.accuracy = 1
        else: # incorrect
            self.score = (1 - min(1, max(self.patient.death, ((self.patient.death + 1)/self.patient.b))))*x
            self.accuracy = -1
        self.speed = self.patient.time
        self.pdeath = self.patient.death
        
    def test(self):
        return np.random.binomial(1, self.patient.truth) # run test (coin toss)
    def update(self, result):
        pass
    def check_stake(self):
        pass
    def step(self):
        if self.active:
            self.update(self.test())
            for i in range(len(self.cred)):
                if self.cred[i] > self.threshold:
                    self.intervene(self.hyp[i])
                    self.active = False
                    break



class FullPure(Doctor):
    def __init__(self, unique_id, model, patient=None, threshold=0.9, N_hyp=11):
        super().__init__(unique_id, model, patient, threshold, N_hyp)
    def update(self, result): # Bayesian
        Pr_E_H = np.absolute((1-result)-self.hyp)
        self.cred = Pr_E_H*self.cred/np.sum(self.cred*Pr_E_H)


class SmallPure(FullPure):
    def __init__(self, unique_id, model, patient=None, threshold=0.9, N_hyp=11, dsize=4):
        super().__init__(unique_id, model, patient, threshold, N_hyp)
        self.dsize = dsize
    def check_stake(self):
        self.contract()
    def contract(self):
        c, h = zip(*sorted(zip(self.cred, self.hyp), reverse=True)[:self.dsize]) # Pick most likely hyp
        self.cred = list(c)/sum(list(c)) # normalize
        self.hyp = list(h)

# Sensitive agent stake normalization
s_mu = 3
s_sigma = 2

class FullSens(FullPure):
    def __init__(self, unique_id, model, patient=None, threshold=0.9, N_hyp=11):
        super().__init__(unique_id, model, patient, threshold, N_hyp)
        self.default = threshold
        self.sensitivity = round(max(min(np.random.normal(loc=0.5, scale=0.1), 1), 0), 2) # random normal truncated [0, 1]
    def check_stake(self):
        self.threshold = round((self.default + (1-self.default)*self.sensitivity*(self.patient.stake-s_mu)/s_sigma), 3)




class cos(SmallPure):
    """SmallSens with threshold one"""
    def __init__(self, unique_id, model, patient=None, N_hyp=11, dsize=4):
        super().__init__(unique_id, model, patient, 1, N_hyp, dsize)
        self.default = dsize
    def check_stake(self):
        self.dsize = min((self.dsize-s_mu+self.patient.stake), self.N_hyp) # size of domain
        self.contract()
    def step(self):
        if self.active:
            self.update(self.test())
            for i in range(len(self.cred)):
                if self.cred[i] == self.threshold: # threshold = 1
                    self.intervene(self.hyp[i])
                    self.active = False
                    break



class SmallSens(SmallPure):
    """SmallSens with fixed threshold less than one"""
    def __init__(self, unique_id, model, patient=None, threshold=0.9, N_hyp=11, dsize=4):
        super().__init__(unique_id, model, patient, threshold, N_hyp, dsize)
    def check_stake(self):
        self.dsize = min((self.dsize-s_mu+self.patient.stake), self.N_hyp)
        self.contract()



"""Data collection functions for doctors only"""
def get_type(agent):
    return type(agent).__name__
def get_stake(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.patient.stake
def get_score(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.score
def get_N_hyp(agent):
    return agent.model.N_hyp
def get_accuracy(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.accuracy
def get_speed(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.speed
def get_truth(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.patient.truth
def get_hyp(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.hyp
def get_a(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.patient.a
def get_b(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.patient.b
def get_scale(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.patient.scale
def get_shape(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.patient.shape
def get_death(agent):
    if type(agent).__name__=="Patient":
        return np.nan
    else:
        return agent.pdeath


# In[131]:


class epistemicModel(Model):
    def __init__(self, N_FullPure=25, N_SmallPure=25, N_FullSens=25, N_cos=25, N_SmallSens=25, threshold=0.9, N_hyp=11, dsize=4):
        self.N_hyp = N_hyp # Total number of hypotheses
        self.N_FullPure = N_FullPure
        self.N_SmallPure = N_SmallPure 
        self.N_FullSens = N_FullSens
        self.N_cos = N_cos
        self.N_SmallSens = N_SmallSens
        self.limit = 100 # time limit before forced termination
        self.dsize = dsize # standard domain size for small domain agents
        self.threshold = threshold # standard threshold for those with fixed threshold
        self.schedule = RandomActivation(self)
        self.total = N_FullPure+N_SmallPure+N_FullSens+N_cos+N_SmallSens # total num of doctors
        
        self.reset()
        
        self.datacollector = DataCollector(
            agent_reporters={"type": get_type,
                             "stake": get_stake, 
                             "score": get_score, 
                             "N_hyp": get_N_hyp, 
                             "accuracy": get_accuracy, 
                             "speed": get_speed, 
                             "truth": get_truth, 
                             "hyp": get_hyp, 
                             "a": get_a, "b": get_b, "scale": get_scale, "shape": get_shape, 
                             "death": get_death
                            }
        )
        self.running = True
        
    def reset(self):
        """Create new agents"""
        k = self.N_FullPure
        l = k+self.N_SmallPure
        m = l+self.N_FullSens       
        n = m+self.N_cos
        o = n+self.N_SmallSens
        
        self.schedule = RandomActivation(self)
        
        # Create agents
        for i in range(k):
            self.schedule.add(FullPure(i, self, None, self.threshold, self.N_hyp))
        for i in range(k, l):
            self.schedule.add(SmallPure(i, self, None, self.threshold, self.N_hyp, self.dsize))
        for i in range(l, m):
            self.schedule.add(FullSens(i, self, None, self.threshold, self.N_hyp))
        for i in range(m, n):
            self.schedule.add(cos(i, self, None, self.N_hyp, self.dsize))
        for i in range(n, o):
            self.schedule.add(SmallSens(i, self, None, self.threshold, self.N_hyp, self.dsize))
        
        # Create patients
        for agent in self.schedule.agents:  
            p = Patient((self.total+agent.unique_id), self, agent, stake=random.choice(range(1, 6)))
            agent.patient = p
            agent.check_stake()
            self.schedule.add(p)

    def step(self):
        for i in range(self.limit):
            self.schedule.step()
        for a in self.schedule.agents:
            if a.active:
                a.intervene() # forced termination after 100 steps
        self.datacollector.collect(self)
        self.reset()







