## The RL Problem

### basic terms

A **reward** $R_t$ is a **scalar** feedback signal

Goal: select actions to maximise total future reward



<img src="image\Snipaste_2023-08-04_18-33-06.png" alt="image-20230804182501338 " style="zoom:70%;"/>

The **history** is the sequence of observations, actions, rewards 
$$
H_t = O_1, R_1, A_1, ..., A_{t−1}, O_t , R_t
$$
**State** is the information used to determine what happens next
$$
S_t = f(H_t)
$$


The **environment state** $S^e_t$ isthe environment’s private representation

The **agent state** $S^a_t$ a t is the agent’s internal representation

An **information state** (a.k.a. Markov state) contains all useful information from the history



A state $S_t$ is <font color=red>Markov</font> if and only if
$$
P[S_{t+1}|S_t]=P[S_{t+1}|S_1,...,S_t]
$$
The state is a sufficient statistic of the future



### observability

Full observability: agent directly observes environment state
$$
O_t =S^e_t=S^a_t
$$
Formally, this is a **Markov decision process** (MDP)



Partial observability: agent indirectly observes environment

Formally this is a **partially observable Markov decision process** (POMDP)



### Major Components of an RL Agent

Policy:

- A policy is the agent’s behaviour 
- It is a map from state to action, e.g. 
- Deterministic policy: $a = π(s) $
- Stochastic policy: $π(a|s) = P[A_t = a|S_t = s]$



Value Function:

- Value function is a prediction of future reward

- Used to evaluate the goodness/badness of states 

- And therefore to select between actions, e.g. 
  $$
  v_π(s) = E_π[ R_{t+1} + γR_{t+2} + γ^2R_{t+3} + ... | S_t = s]
  $$

Model:

- A model predicts what the environment will do next 

- Transitions: P predicts the next state 

- Rewards: R predicts the next (immediate) reward, e.g. 
  $$
  P^a_{ss'} = P[S_{t+1} = s' | S_t = s, A_t = a] \\
  R^a_ s = E [R_{t+1} | S_t = s, A_t = a]
  $$

### Categorizing

<img src="image\Snipaste_2023-08-04_18-49-45.png" alt="Snipaste_2023-08-04_18-49-45" style="zoom:50%;" />



## Markov Decision Processes

A **Markov Decision Process** is a tuple $<\mathcal S,\mathcal A,\mathcal P,\mathcal R, \gamma>$

- S is a infite set of states
- A is a infite set of actions
- P is a state transition probability matrix, $\mathcal P^a_{ss'} = \mathbb P [S_{t+1} = s'|S_t = s,A_t = a]$
- R is a reward function, $\mathcal R^a_s = \mathbb E[R_{t+1}|S_t=s, A_t=a]$
- $\gamma$ is a discount factor $\gamma \in [0,1] $



A **policy** $\pi$  is a distribution over actions given states,
$$
\pi(a|s)= \mathbb P[A_t=a|S_t=s]
$$

- A policy fully denes the behaviour of an agent
- Policies are stationary (time-independent)
- Reward is given in the S



The **state-value function** $v_\pi(s)$ of an MDP is the expected returns tarting from state s, and then following policy $\pi$
$$
v_\pi(s)=\mathbb E_\pi[G_t|S_t=s]=\mathbb E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})|S_t=s]
$$
The **action-value function** $q_\pi(s, a)$ is the expected return starting from state s, taking action a, and then following policy $\pi$
$$
q_\pi(s,a)=\mathbb E_\pi[G_t|S_t=s,A_t=a]
$$

### Bellman Expectation Equation

$$
q_\pi(s,a) = \mathcal R^a_s+\gamma\sum_{s'\in \mathcal S}\mathcal P^a_{ss'}v_\pi(s')\\
=\mathcal R^a_s+\gamma\sum_{s'\in \mathcal S}\mathcal P^a_{ss'}\sum_{a'\in\mathcal A}\pi(a'|s')q_\pi(s',a')
$$

### Optimal Value Function and Policy

The optimal state-value function $v_*(s)$ is the maximum value function over all policies
$$
v_*(s) = \underset {\pi}{max}\ v_\pi(s)
$$
The optimal action-value function $q_*(s, a)$ is the maximum action-value function over all policies
$$
q_*(s, a)=\underset {\pi}{max}\ q_\pi(s,a)
$$
Define a partial ordering over policies
$$
\pi \geq \pi' \quad if\quad  v_\pi(s)\geq v_{\pi'}(s), \forall s
$$
**For any Markov Decision Process**:

- There exists an **optimal policy**  $\pi_*$that is better than or equal to all other policies, $\pi_* \geq\pi,\forall \pi$
- All optimal policies achieve the optimal value function, $v_{\pi_*}(s)=v_*(s)$
- All optimal policies achieve the optimal action-value function,$q_{\pi_*} (s,a)=q_*(s, a) $



An optimal policy can be found by maximising over$q_*(s, a)$,
$$
\pi_*(a|s)=\begin{cases}
1\quad if\ a=\underset{a\in\mathcal A}{argmax}q_*(s, a)\\
0\quad otherwise
\end{cases}
$$

- There is always a **deterministic optimal policy** for any MDP
- If we know $q_*(s, a)$, we immediately have the optimal policy

### Bellman Optimality Equation

$$
q_*(s,a) =\mathcal R^a_s+\gamma\sum_{s'\in \mathcal S}\mathcal P^a_{ss'}\underset{a'}{max}\ q_*(s',a')
$$

- Bellman Optimality Equation is non-linear
- No closed form solution (in general)
- Many iterative solution methods
  - Value Iteration
  - Policy Iteration
  - Q-learning
  - Sarsa

## Planning by DP



### Policy Evaluation

Problem: evaluate a given policy $\pi$
$$
v_{k+1}(s)=\sum_{a\in\mathcal A}\pi(a|s)(\mathcal R^a_s+\gamma\sum_{s'\in \mathcal S}\mathcal P^a_{ss'}v_k(s'))
$$

### Policy improvement

Given a policy 

​	Evaluate the policy 
$$
v_π(s) = \mathbb E[ R_{t+1} + γR_{t+2} + γ^2R_{t+3} + ... | S_t = s]
$$
​	Improve the policy by acting greedily with respect to $v_\pi$
$$
\pi'=greedy(v_\pi)
$$
this process of policy iteration always **converges to $\pi*$**

### Value Iteration

Problem: nd optimal policy $\pi$

Solution: iterative application of Bellman optimality backup

Using synchronous backups

- At each iteration k + 1
- For all states $s\in\mathcal S$
- Update $v_{k+1}(s)$ from $v_k(s')$

$$
v_{k+1}(s)=\underset{a\in\mathcal A}{max}\ (\mathcal R^a_s+\gamma\sum_{s'\in \mathcal S}\mathcal P^a_{ss'}v_k(s'))
$$

## Model-Free Prediction

### Monte-Carlo Learning

Goal: learn v from **episodes of experience** under policy $\pi$

Update V(s) incrementally after episode S1; A1; R2;...; ST
For each state St with return Gt
$$
N(s_t)\leftarrow N(S_t)+1\\
V(S_t)\leftarrow V(S_t)+\frac 1 {N(s_t)}(G_t -V(S_t))\\
$$

### Temporal-Difference Learning

TD learns from incomplete episodes

TD(0):
$$
V(S_t)\leftarrow V(S_t)+\alpha(R_{t+1}+\gamma V(S_{t+1}) -V(S_t))
$$
$R_{t+1}+\gamma V(S_{t+1})$ is called the TD target

$\delta_t = R_{t+1}+\gamma V(S_{t+1}) -V(S_t)$ is called the TD error



$G_t$ is *unbiased* stimate of $v_π(S_t)$

TD target is *biased* stimate of $v_π(S_t)$

TD target is much lower variance, usually more efficient than MC



### TD($\lambda$)

Forward-view TD($\lambda$):
$$
G_t^{(n)} = R_{t+1}+\gamma R_{t+2}+...+\gamma^nV(S_{t+n})\\
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infin}\lambda^{n-1}G_t^{(n)}\\
V(S_t)\leftarrow V(S_t)+\alpha(G_t^\lambda -V(S_t))
$$
Backward-view TD($\lambda$):

Update value V(s) for every state s

### eligibility traces

<img src="image\image-20230818202320718.png" alt="image-20230818202320718" style="zoom:60%;" />

<img src="D:\课外\ai\RL\image\image-20230818202406508.png" alt="image-20230818202406508" style="zoom:50%;" />

## Model-Free Control

### $\epsilon$-Greedy Exploration

$$
\pi(a|s)=
\begin{cases}
\epsilon/m+1-\epsilon\quad if\ a^*=\underset{a\in\mathcal A}{argmax}Q(s|a)\\
\epsilon/m \quad otherwise
\end{cases}
$$

### Sarsa

Saras(0):

<img src="image\image-20230818105234950.png" alt="image-20230818105234950" style="zoom:50%;" />

Sarsa($\lambda$): familiar with TD($\lambda$) , includes Forward-view and Backward-view

### Off-Policy Learning

Evaluate target policy $\pi(a|s)$ to compute v(s) or $q_\pi (s, a)$
While following **another** behaviour policy $\mu(a|s)$

TD:

Use TD targets generated from  $\mu$ to evaluate $\pi$
$$
V(S_t)=V(S_t)+\alpha(\frac{\pi(A_t|S_t)}{\mu(A_t|S_t)} (R_{t+1}+\gamma V(S_{t+1}))-V(S_t))
$$
Policies only need to be similar over a single step



Q-Learning:

off-policy learning of action-values

$A'$ is $\mu$

<img src="image\image-20230818201343797.png" alt="image-20230818201343797" style="zoom:50%;" />

### Off-Policy Control with Q-Learning

The target policy  is greedy give briefcase_of_cash

The behaviour policy is $\epsilon$-Greedy

<img src="image\image-20230818201645777.png" alt="image-20230818201645777" style="zoom:50%;" />

## Function Approximation

### Incremental Methods

<img src="image\image-20230819222308090.png" alt="image-20230819222308090" style="zoom:70%;" />

<img src="image\image-20230819222402048.png" alt="image-20230819222402048" style="zoom:67%;" />

### Batch Methods

Deep Q-Networks:

<img src="image\image-20230819222537077.png" alt="image-20230819222537077" style="zoom:67%;" />



Linear Least Squares Prediction:

<img src="image\image-20230819222616251.png" alt="image-20230819222616251" style="zoom:67%;" />

## Policy Gradient Methods

parametrise the policy
$$
\pi_\theta(s,a)=\mathbb P[a|s,\theta]
$$
Advantages of Policy-Based RL:

- Better convergence properties
- Eective in high-dimensional or continuous action spaces
- Can learn stochastic policies

Disadvantages:

- Typically converge to a local rather than global optimum
- Evaluating a policy is typically inecient and high variance

### Finite Dierences

<img src="image\image-20230824183743075.png" alt="image-20230824183743075" style="zoom: 67%;" />

### Score Function

<img src="image\image-20230824200702955.png" alt="image-20230824200702955" style="zoom:67%;" />

<img src="image\image-20230824200843729.png" alt="image-20230824200843729" style="zoom:67%;" />

For multi-step MDPs, replaces instantaneous reward r with long-term value Q(s, a):
$$
\nabla J(\theta)=\mathbb E_{\pi_\theta}[\nabla_\theta \ log\,\pi_\theta(s,a)\ Q^{\pi_\theta}(s,a)]
$$

### Monte-Carlo Policy Gradient (REINFORCE)

<img src="image\image-20230824201328562.png" alt="image-20230824201328562" style="zoom:67%;" />

high variance, slow



### Actor-critic

use a critic to estimate the action-value function $Q_w(s,a)$

<img src="image\image-20230824201501391.png" alt="image-20230824201501391" style="zoom:67%;" />

### Advantage Function

We subtract a baseline function B(s) from the policy gradient
This can reduce variance, without changing expectation

<img src="image\image-20230824202054733.png" alt="image-20230824202054733" style="zoom:67%;" />

we can use two function approximators and two parameter vectors

or:

<img src="image\image-20230824205546068.png" alt="image-20230824205546068" style="zoom: 67%;" />



## Integrating Learning and Planning

### Model-Based Learning

<img src="image\image-20230827214540556.png" alt="image-20230827214540556" style="zoom:67%;" />

Model learning:

​	estimate model $M_\eta$ from experience: **supervised learning problem**

for example:

<img src="image\image-20230827214904010.png" alt="image-20230827214904010" style="zoom:50%;" />

then planning with a  model

Sample-Based Planning:

- Use the model only to generate samples
- Sample experience from model
- Apply model-free RL to samples
- Sample-based planning methods are often more ecient

### Integrated Architectures

<img src="image\image-20230827215125952.png" alt="image-20230827215125952" style="zoom:67%;" />

use both real exp and simulated exp:

Dyna-Q learning:

<img src="image\image-20230827215234389.png" alt="image-20230827215234389" style="zoom:50%;" />

Dyna-2:

<img src="image\image-20230827221101755.png" alt="image-20230827221101755" style="zoom:67%;" />

### Simulation-Based Search

1. Simple Monte-Carlo Search

<img src="image\image-20230827220132801.png" alt="image-20230827220132801" style="zoom:67%;" />

2. Monte-Carlo Tree Search (Evaluation)

<img src="image\image-20230827220300368.png" alt="image-20230827220300368" style="zoom:67%;" />

this method improves policy  e.g.  $\epsilon$-Greedy

*Monte-Carlo control* applied to simulated experience



3. TD search

TD search applies Sarsa to sub-MDP from now

<img src="image\image-20230827221004659.png" alt="image-20230827221004659" style="zoom:67%;" />Dyna-2
