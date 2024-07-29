# QLearning (QL) and Deep Q Learning (DQL) applied to the maze problem

## Why this Repository

I created this repository to explore and implement a Reinforcement Learning (RL) algorithm by myself. The theoretical part of RL and the two algorithms are not covered here; instead, I focus on providing a clear implementation of these algorithms. If you're looking for a more theoretical explanation, I invite you to follow the work this repository is based on!

If you find any problems in the implementation or README, feel free to inform me.

## The algorithms for the maze problem

In this repository, you will find a program that explores the QL and DQL algorithms applied to the maze problem.
This problem is simple: we create a maze (an environment) with some walls, a rat (an agent), and a cheese (the goal of the agent).
The purpose of the rat is to navigate through the maze to reach the cheese.

At each state, we provide information to the agent about its current state. The agent takes an action and receives a reward from the environment based on that action. But how does the agent make a choice? That's the goal of the QL and DQL algorithms.

### QLearning

To make a choice, we need to define a policy for the Agent, here called Q. This policy is a table where, for each state, we have a value for each action. If $ N $ is the number of states and $ M $ is the number of actions, we obtain a table of size $ (N \times M) $. When we create our rat, this policy is initialized with random values. Our goal is to improve this policy to become more accurate regarding the current maze.

What are the states? In the QL algorithm, the states are only a tuple of the current position of the rat $ (i, j) $.

What are the actions? We assume the rat can move in four directions: up, down, right, and left.

Now, there is a maze and a rat. At this point, the rat can move in the maze but will probably not end up at the cheese. First of all, how does the rat make a choice in the implementation? It's quite simple: for each state, it looks for the maximum value in the Q table and takes the action with the highest Q value for that state. However, at this point, if we only do that, we have two problems:
- We are not sure that with the initialized values of Q, the rat will make the best choice. We need to explore the maze.
- The values of Q are fixed; we would like to modify them based on the results.

To resolve the exploration problem, we set a value $ \epsilon $ between 0 and 1, which represents the exploitation chance/percentage, and $ 1 - \epsilon $ represents the exploration chance/percentage. So, at each state, we either make an exploitation choice (following the table) or an exploration choice (taking a random action).

To resolve the Q problem, the QL algorithm comes into play. At each state, the agent takes an action and gets a reward. With this value, we can modify the line of the Q table for the current state. I will not go into more detail on the theoretical part, but you can see [2] for the mathematical details. The key point to retain is that the lower the reward, the lower the Q value for that state and action will be, and thus, a lower chance of being chosen during an exploitation choice. The agent is learning from its actions.

At this point, we have everything necessary to make the rat solve the maze. Now, it only needs to learn the maze. To do this, we run many iterations of this problem to make the rat learn.

A good way to improve this algorithm is to make the $ \epsilon $ value change with the iteration. In fact, at the start, we know that the policy is not accurate, so we need to prioritize exploration. At the opposite end, when the policy may be accurate, we want to reduce the exploration choice. A good way to do that is to decrease the $ \epsilon $ value over the iterations.

#### The code

If you want to launch the QL algorithm, go to the main file and set the `game` variable to the `rat` variable.

### Deep QLearning

The principle of the DQL algorithm is the same as the QL one. The only difference is the management of the policy (Q table) and the state variable. As the term "deep" suggests, we use neural networks for the Q table.

The state variable changes to provide more information and more inputs to the neural networks. We provide a list of the maze, with 0s and 1s representing free cells/walls and the `rat_mark` value where the rat is.

Now, at each step, if we're doing exploitation, we input this state into the neural network, which gives us the line of the Q values for the current state, and it chooses the highest one as its action.

#### The code

If you want to launch the DQL algorithm, go to the main file and set the `game` variable to the `neuralRat` variable.

# Documentation

As mentioned in a previous section, this work is heavily inspired by three sources:
[1]: https://www.samyzaf.com/ML/rl/qmaze.html
-> Useful for the global implementation, including the environment, the maze, and the QL algorithm.
[2]: https://fr.wikipedia.org/wiki/Q-learning
-> Useful for the theoretical part and a better understanding.
[3]: https://github.com/giorgionicoletti/deep_Q_learning_maze/tree/master
-> Useful for the implementation of the DQL algorithm with the neural network.