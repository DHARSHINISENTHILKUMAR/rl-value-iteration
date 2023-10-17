# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## State
## 5 Terminal states
G - (Goal): The state the agent aims to reach. H - (Hole): A hazardous state that the agent must avoid at all costs.
## 11 Non-Terminal States:
S - (Starting state): The initial position of the agent. Intermediate states: Grid cells forming a layout that the agent must traverse.
## Actions:
The agent can take 4 actions in each state:
LEFT RIGHT UP DOWN

## Transition Probabilities:
The environment is stochastic, meaning that the outcome of an action is not always certain.

33.33% chance of moving in the intended direction. 66.66% chance of moving in a orthogonal directions. This uncertainty adds complexity to the agent's navigation.

## Rewards:
+1 for reaching the goal state(G). 0 reward for all other states, including the starting state (S) and intermediate states. Episode Termination: The episode terminates when the agent reaches the goal state (G) or falls into a hole (H)

## Graphical Representation:
![image](https://github.com/DHARSHINISENTHILKUMAR/rl-value-iteration/assets/113699377/43fc7147-6a52-46cb-9c05-2ac5cfeae8f7)


## POLICY ITERATION ALGORITHM
->Value iteration is a method of computing an optimal MDP policy and its value. ->It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation. ->The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.

The algorithm is as follows:

Initialize the value function V(s) arbitrarily for all states s.
Repeat until convergence: ->Initialize aaction-value function Q(s, a) arbitrarily for all states s and actions a. ->For all the states s and all the action a of every state: 1.Update the action-value function Q(s, a) using the Bellman equation. 2.Take the value function V(s) to be the maximum of Q(s, a) over all actions a. 3.Check if the maximum difference between Old V and new V is less than theta. 4.Where theta is a small positive number that determines the accuracy of estimation.
If the maximum difference between Old V and new V is greater than theta, then Update the value function V with the maximum action-value from Q. Go to step 2.
The optimal policy can be constructed by taking the argmax of the action-value function Q(s, a) over all actions a.
Return the optimal policy and the optimal value function.

## VALUE ITERATION FUNCTION
```
REG NO:212221220009
NAME: DHARSHINI S
```
```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if(np.max(np.abs(V-np.max(Q,axis=1))))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s , a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```
## OUTPUT:

## Optimal Policy:
![image](https://github.com/DHARSHINISENTHILKUMAR/rl-value-iteration/assets/113699377/1f962014-2180-45b6-a262-22b2929bce42)
## Optimal Value Function:
![image](https://github.com/DHARSHINISENTHILKUMAR/rl-value-iteration/assets/113699377/8f0647bb-eb04-424f-bfcc-199b227ab553)
## Success Rate for Optimal Policy:
![image](https://github.com/DHARSHINISENTHILKUMAR/rl-value-iteration/assets/113699377/51c2c21d-80b7-4b29-901e-b71b15b420bd)


## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
