import numpy as np

import matplotlib.pyplot as plt



def generateAdjacencyMatrix(n, type="random", p=0.1, rng=np.random.default_rng()):
    match type:
        case "random":
            A = rng.random((n,n));
            A = A>p;
            A = A+A.T;
            A = A - (A==2).astype(float);
            A = A - np.diag(np.diag(A))
        case "cycle":
            A = np.diag(np.ones([1,n-1])[0],-1)+np.diag(np.ones([1,n-1])[0],1);
            A[n-1,0] = 1;
            A[0,n-1] = 1;
        case other:
            A = np.zeros([n,n]);
    return A


#Parameters
n= 10; #Number of households
I = np.eye(n);
k = 0;
max_iter = 99;
influence = 0.001;
learning = 0.01;
agreement = 0.001;
lbd = 0;
threshold_close = 0.5;
threshold_far = 1;
rng = np.random.default_rng(seed=None)

Understanding = rng.choice([-0.75,0.1, 0.75], [n,1], [1/3,1/3,1/3]);
SocialOpinions = ( rng.random([n,1])-0.5 ) * 0.1;
Opinions = np.clip(SocialOpinions+Understanding,-1,1);


current_dispersion = np.max(np.abs(SocialOpinions @ np.ones((1,n)))-SocialOpinions @ np.ones((1,n)));

Trajectory_Opinions = [Opinions[:,0]];
Trajectory_social_opinions = [SocialOpinions[:,0]];
Trajectory_Understanding = [Understanding[:,0]];

while (current_dispersion>agreement)& (k < max_iter):
    k += 1;
    #Create the network
    L = (np.abs(Opinions @ np.ones([1,n]) - (Opinions @ np.ones([1,n])).T ) < threshold_close) - np.eye(n);
    Lclose = np.diag(np.sum(L,1))- L;

    L = (np.abs(Opinions @ np.ones([1,n])- (Opinions @ np.ones([1,n])).T ) > threshold_far) - np.eye(n);

    Lfar = np.diag(np.sum(L,1))- L;

    L = generateAdjacencyMatrix(n,'random', 1-lbd, rng);
    Lalea = np.diag(np.sum(L,1))- L;

    L = Lclose - Lfar + Lalea;

    #Update social opinions
    SocialOpinions = (I - influence * L) @ Opinions - Understanding;

    if k%1==0:
        Understanding = np.clip(Understanding + learning, -1,1);

    Opinions = np.clip(SocialOpinions+Understanding,-1,1);
    current_dispersion = np.max(np.abs(SocialOpinions @ np.ones((1,n)))-SocialOpinions @ np.ones((1,n)));

    Trajectory_Opinions += [Opinions[:,0]];
    Trajectory_social_opinions += [SocialOpinions[:,0]];
    Trajectory_Understanding += [Understanding[:,0]];


plt.plot(np.arange(0,k+1,1), Trajectory_Opinions, ':+')
plt.plot(np.arange(0,k+1,1), Trajectory_Understanding, 'k-')
plt.xlabel("Timesteps")
plt.ylabel("Support level (-1 is fully opposed, +1 fully supportive)")
plt.title("Coupling understanding and social learning on opinions dynamics")
plt.legend(["Agents opinions"]+[""for iu in range(n-1)]+["Understandings","", ""],bbox_to_anchor=(1.02,1), loc='upper left')
plt.subplots_adjust(right=0.8)
plt.show()

