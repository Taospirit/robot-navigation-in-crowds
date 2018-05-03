# Robot-navigation-in-crowds

In this project I build up a virtual 2d environment and use reinforcement learning to teach a mobile robot to avoid obstacles and reach the goal.  The objective of the problem is to minimize the path_length of the robot.

## 1. Set up

I develop this project using Python3, Pygame, Pymunk, and Keras. The Pymunk module is imported to simulate the physics environment and Pygame module is imported to draw all the stuff on the screen. Keras is used for building up the network model.

### Installation

    $ pip install pygame
    $ pip install pymunk
    $ pip3 install tensorflow
    $ pip install keras
    $ pip install h5py

## 2. Problem Statement

### Environment

I build up the GameClass.py that contains a class to simulate all the objects and behaviours in the simulation environment, and run a simple trial problem in it. At each time step, the `frame_step()` function will run and return the state and reward parameter. A 2D rectangle space with width = 1200 and height = 900 is constructed to simulate the envirionment. Every object in the environment is constructed via pymunk.Body() method and its shape is set using the constructed body.

### State space

Suppose the position of the robot is $p_1$ and $p_2$. The pointing angle of the robot is $\theta$. The robot have 3 action choices at each time frame, and they are represented by number, which is: 

- 0: turn right 5 degree.
- 1: turn left 5 degree.
- 2: do nothing.

The state of the robot is a 3 elements vector. which is $[p_1,p_2,\theta]$.

### paremeters

| Name      | Description           | Value     |
| -         | -                     |- |
| EPISODE   | Total rounds to play  | 10 |
| FRAMES    | Total frames to play in a round   | 4000 |
| OBSERVE   | Number of frames to observe before trainning  | 12000 |
| epsilon   | The parameter for the $\epsilon$-greedy algorithm | 1 |
| GAMMA     | The discount factor   |0.9    |
| batchSize | Number of trainning records the network used to update it's value.    |40, 100|
| bufferSize    | The length of the records saved in the memory.    | 10000 |
| NUM_INPUT | Number of input values to the neural network. |   6   |
|   FPS | Frames per second, used in the game environment.  |   60  |
| hidden layer parameters | The number of neurons in each of the two hidden layers. |(128,128),(256,256),(512,512) |

## 3. Q-Learning algorithm

Here I combined Q-learning with experience replay and a simply 2 hidden layers neural network. Q-Learning algorithm has several benefits: model-free, so it doesn't require the environment model and thus computationally cheaper; it is a temporal difference learning altorithm. So it can update the expectation of the accumulated reward incrementally instead of accumulating the reward value. For this problem, the algorithm looks like this:
    1. Initialize parameters. Set total frames $tf=0$.
    2. Initialize  the neural network `model` and set the number of neurons in each hidden layer.
    3. **For** m = 1, ..., EPISODE, **do**
    4.     Initialize a game object and start the simulation. Get the initial state $s_1$ and reward $r_1$.
    5.     **For** t = 1, ..., FRAMES, **do**
    6.         count total frames $tf=tf+1$.
    7.         With probability epsilon, or if $tf$ < OBSERVE,  select a random action $a_t$.
    8.         Otherwise select $a_t = max_a Q(s_t, a)$
    9.         Execute action $a_t$ in the simulator and observe reward $r_t$ and new state $s_{t+1}$.
    10.        Store the pair $(s_t,a_t, r_t, s_{t+1})$ in the memory $M$.
    11.        **If** $tf$ > OBSERVE, **do**
    12.            Sample random minibatch pairs $(s_j,a_j, r_j, s_{j+1})$ from $M$.
    13.            For each sample in the minibatch:
    14.                one-hot encode the action $a_j$.
    15.                Combine the state and action as the features: $f_j = [s_j,\text{encoded-a}_j]$ and append it in **X_train**. 
    16.                Set value function prediction for $Q(s_j, a_j)$ as: 
    17.                $$ y_j=     \begin{cases} r_j, & \text{for terminal } s_j\\r_j + \gamma \max_{a'} Q'(s_j+1, a'), & \text{for non-terminal } s_j   \end{cases}$$
    18.                Append the $y_j$ to **y_train**.
    19.            Use **X_train** and  **y_train** to train the model. 
    20.        **End If**
    21.        Update the state $s_{t+1} = s_t$.
    22.    **End For**
    23. **End For**
    24. Output: the `model`

The one-hot encoding of the action $a_j$:
$$ \text{encoded-a}_j =    \begin{cases}
[1,0,0], & a_j = 0 \\
[0,1,0], & a_j = 1 \\
[0,0,1], & a_j = 2 \\
\end{cases}$$
### Reward
The objective of the problem is to maximize the earned reward. The reward earned by executing action $a$ at state $s$ is set as:

$$R(s,a) = -\text{runtime}-50 \cdot \mathbb{1}_{[\text{hit an obstacle}] } - 50 \cdot \mathbb{1}_{[\text{hit a wall}] } + 10000 \cdot \mathbb{1}_{[\text{reach the goal}] }$$

Since $Q(s,a)$ is the average of $R(s,a)$ over all the states and actions after $(s,a)$, every time $a_t = max_a Q(s_t, a)$ will optimize the Q value.

## 4. Neural network structure

The neural network is built up in the `nn.py` file. It is used to predict the value of $Q(s,a)$. The structure of the network is illustrated below:

![nn_model](/nn_model.png)

The neural network model has two hidden layer. The first layer has input size of 6, which is equal to the size of the feature. In initialization I set a LeCun normal initializer and the bias initialize to zero. The activation function is ReLU for all the layers except the output layer, which is linear. To prevent overfit, dropout is added. In compilation, I use RMSprop optimizer and mean square loss as loss function.

## Experiments
### Tranning process
As the number of hidden neurons per layer or the batch grows, the training speed goes down rapidly. So I set the parameters as above and begin trainning. The loss log for each frame is illustrated here:

![loss-128](/results/logs/loss_data-128-128-100-10000-9.csv.png)
![loss-256](/results/logs/loss_data-256-256-100-10000-9.csv.png)
![loss-512](/results/logs/loss_data-512-512-100-10000-9.csv.png)

Explanation:
The name 128-128-100-10000-9 means the network has 128 neurons at first layer and 128 neurons at second layer. The batch size is 100 and the buffer size is 10000. 9 means it is recorded after 10 models(start from 0).

The loss data has huge oscillation. when the network size is 128 and 256, the loss doesn't decreace or converge, which means poor trainning performance. When the network size is 256, it converge to a lower value, so it trains well.

I also record the path length of the robot if it reach the goal in a training episode. They are illustrated here:

![path-128](/results/logs/path_data-128-128-100-10000-9.csv.png)
![path-256](/results/logs/path_data-256-256-100-10000-9.csv.png)
![path-512](/results/logs/path_data-512-512-100-10000-9.csv.png)

The robot will reach the goal 4 or 5 times in a trainning progress. The average length decrease when the neural network get more complicated, which means a more complex model represents the environment better.

### Testing process
I record the model after every training episode and use these models for testing. In the test, the robot's movement is totally decided by the Q value, which means the policy is $a_t = max_a Q(s_t, a)$ for each action. The results of the testing is shown below:

- test with model 128-128-100-10000-9

![test-128](results\test\128-128-100-10000.png)

- test with model 256-256-100-10000-9

![test-256](results\test\128-128-100-10000.png)

- test with model 512-512-100-10000-9

![test-512](results\test\128-128-100-10000.png)

