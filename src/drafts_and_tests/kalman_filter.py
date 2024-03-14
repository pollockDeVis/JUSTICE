import numpy as np 
import scipy as sci
import pandas as pd

def parameters_to_kalman(gamma, C, kappa, epsilon, sigma_eta, sigma_xi, F_4xC02):
    k = len(C);
    if (k==2):
        A = np.array([[-gamma, 0, 0],\
                      [1/C[0]],-(kappa[0] + epsilon*kappa[1])/C[0], epsilon*kappa[1]/C[0]],\
                      [0, kappa[1]/C[1], -kappa[1]/C[1]]);   
    elif (k==3):
        A =np.array([[-gamma, 0, 0, 0],\
            [1/C[0], -(kappa[0] + kappa[1])/C[0], kappa[1]/C[0], 0],\
            [0, kappa[1]/C[1], -(kappa[1] + epsilon*kappa[2])/C[1], epsilon*kappa[2]/C[1]],\
            [0, 0, kappa[2]/C[2], -kappa[2]/C[2]]]);
    else: 
        raise Exception("Dimension of C shoud be 2 or 3 but is ", k, "!")
        
    
    b = np.matrix([gamma]+[ 0 for i in range(k)]).T
    Q = np.diag([sigma_eta**2, (sigma_xi/C[0])**2]+[0 for i in range(k-1)])
    
    k = k+1;
    Ad = sci.linalg.expm(A);
    bd = np.linalg.solve(A, (Ad - np.eye(k)) @ b);
                         
    H = np.block([[-A, Q],[ np.zeros((k,k)), A.T]])
    G = sci.linalg.expm(H);
    Qd = G[k:2*k, k:2*k] @ G[0:k, k:2*k];
    
    Gamma0 = np.linalg.solve(np.eye(k**2) - np.kron(Ad, Ad), np.reshape(Qd, k**2));
    Gamma0 = np.reshape(Gamma0, [k,k]);

    k = len(kappa)
    if (k==2):
        Cd = np.array([[0,1,0],\
                        [1, -kappa[0] + (1 - epsilon) * kappa[1], -(1 - epsilon) * kappa[1]]]);
    elif (k==3):
        Cd = np.array([[0,1,0,0],\
                       [1, -kappa[0], (1-epsilon)*kappa[2], -(1-epsilon)*kappa[2]]]);
    else:
        raise Exception("Dimension of kappa shoud be 2 or 3 but is ", k, "!")
        
    return [Ad, bd, Qd, Gamma0, Cd]
    

class Kalman_Filter():
    def __init__(self, n):
        self.n = n;
        self.x = np.zeros((n, 1));
        self.xpred = np.zeros((n, 1));
        self.A = np.zeros((n,n));
        self.b = np.zeros((n,1));
        self.u = 0
        self.P = np.zeros((n,n));
        self.Ppred = np.zeros((n,n));
        self.Q = np.zeros((n,n)); #HH'
        self.y = np.zeros((n, 1));
        self.err = np.zeros((n, 1));
        self.residual = np.zeros((n, 1));
        self.C = np.zeros((n,n));
        self.K = np.zeros((n,n));
        self.S = np.zeros((n,n));
        self.Sigma = np.zeros((n,n)); #GG'
        self.iv = np.zeros((n,n));
        
        
        self.measurements = [];#Surface temperature T1 and top-of-atmosphere radiative flux N in column, one column per timestep
        
        self.t = 0;
        self.logL = 0;
        
    def fill(self, x0, Ad, Bd, Qd, Gamma0, Cd, F_4xC02, dataset):
        self.x = Ad @ x0 + Bd *  F_4xC02;
        self.P = Gamma0;
        self.b = Bd;
        self.u = np.matrix(F_4xC02);
        
        self.A = Ad;
        self.C = Cd;
        self.Q = Qd;
        self.Sigma = 1e-12*np.eye(2)
        self.S = self.Sigma + self.C @ self.P @ self.C.T;
        
        
        
        self.measurements = dataset;
        
        self.prediction()
        
    def update(self):
        
        self.err = self.y - self.C @ self.xpred;
        self.S = self.Sigma + self.C @ self.P @ self.C.T;
        
        #here we can eventually prevent any updating based on malahanobis distance with err, x and p
        self.K = self.Ppred @ self.C.T @ np.linalg.pinv(self.S);        
        self.x = self.xpred + self.K @ self.err;
        
        I = np.eye(self.K.shape[0]);
        self.iv = (I - self.K @ self.C)
        self.P = self.iv @ self.Ppred @ self.iv.T + self.K @ self.Sigma @ self.K.T;
        
        self.residual = self.y - self.C @ self.x;

    def prediction(self):
        self.xpred = self.A @ self.x + self.b @ self.u;
        self.Ppred = self.A * self.P * self.A.T + self.Q;
                
        
    def step(self):
        self.y = self.measurements[:,self.t];
        self.update()
        self.prediction()
        self.t += 1;
    
    def run(self):
        while self.t < self.measurements.shape[1]:
            self.step()
            print("---- Iteration ", self.t," of ", self.measurements.shape[1], " ----")
            print(self.x, self.y)
        return self.comp_logL()
            
        
    def comp_logL(self):
        self.logL = - self.t/2 * np.log(2*np.pi) - 1/2 * np.sum(np.log(self.S) + np.divide(np.multiply(self.residual, self.residual), self.S))
    
'''
-------------o0o-------------
           EXEMPLE
-------------o0o-------------
'''
#Starting values for parameters
Gamma0 = 2.0;
C0 = np.array([5.,20.,100.]);
Kappa0 = np.array([1.,2.,1.]);
epsilon = 1.0;
sigma_eta = 0.5;
sigma_xi = 0.5;
F_4xC02 = 5.;
x0 = np.matrix([F_4xC02]+[0 for i in range(len(C0))]).T

#data loading
df = pd.read_csv('flux_temp.csv');
dataset = df.get(['BCC.temp', 'BCC.flux']).to_numpy().T

#Compute matrices from parameters
M = parameters_to_kalman(Gamma0, C0, Kappa0, epsilon, sigma_eta, sigma_xi, F_4xC02);
K = Kalman_Filter(2)
K.fill(x0, M[0], M[1], M[2], M[3], M[4], F_4xC02, dataset)
K.run()


























