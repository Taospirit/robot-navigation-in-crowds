from GameClass import GameClass
import numpy as np
import random
import csv
from nn import neural_net, LossHistory
import os.path
import timeit
from keras.utils import plot_model

TUNING = False
GAMMA = 0.9
NUM_INPUT = 6
FPS = 60

def train(model, params):
    filename = params_to_filename(params)

    EPISODE = 10
    FRAMES = 4000
    OBSERVE = FRAMES*3
    epsilon = 1
    batchSize = params['batchSize']
    buffer = params['buffer']
    replay = []
    minibatch = []
    total_frames = 0
    path_log = []
    loss_log = []

    # min_path_length = 0

    for m in range(EPISODE):
        print("Episode: %d" % (m))
        gameObject = GameClass(draw_screen = True, display_path = True, fps = FPS)

        # Choose no action in the initial frame
        action = 2
        reward, state = gameObject.frame_step(action) 
        for t in range(FRAMES):
            total_frames+=1

            if t%(FRAMES/10) == 0:
                print("Frames: %d" % (t))

            # Choose the action based on the epsilon greedy algorithm
            if (random.random() < epsilon or total_frames < OBSERVE):  # choose random action
                action = np.random.randint(0, 3)
            else:  # choose best action from Q(s,a) values
                # Let's run our Q function on (state,action) to get Q values for all possible actions
                Q = np.zeros(3)
                for a in range(3):
                    features = get_features(state,a)
                    Q[a] = model.predict(features, batch_size=batchSize)
                    action = (np.argmax(Q))

            # Execute the action, observe new state and reward
            reward, state_new = gameObject.frame_step(action)
            path_length = gameObject.num_steps

            # Store the (state, action, reward, new state) pair in the replay
            memory = state, action, reward, state_new
            replay.append(memory)

            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)

            # Randomly sample our experience replay memory if we have enough samples
            if total_frames > OBSERVE:
                minibatch = random.sample(replay, batchSize)

                # Process the minibatch to get the training data
                X_train, y_train = process_minibatch(minibatch,model,batchSize)

                # Train the model on this batch.
                history = LossHistory()
                model.fit(X_train, y_train, batch_size=batchSize,verbose=0,callbacks=[history])
                loss_log.append(history.losses)

                # Decrement epsilon over time.
                if epsilon > 0.1:
                    epsilon -= 1.0/(FRAMES*EPISODE-OBSERVE)

            # Update the starting state with S'.
            state = state_new

            # Stop this episode if we achieved the goal
            if gameObject.check_reach_goal():
                # Log the robot's path length
                path_log.append([m,path_length])

                # # Update the min
                # if path_length < min_path_length:
                #     min_path_length = path_length

                # # Output some stuff so we can watch.
                # print("Min: %d \t epsilon %f\t(%d)" %
                #   (min_path_length, epsilon, path_length))

                # Stop this episode
                break
        
        # Save the model every episode after observation.
        if total_frames > OBSERVE:
            model.save('saved-models/model_nn-'+filename+'-'+str(m)+'.h5', overwrite=True)
            print("Saving model %s - %d" % (filename, m))

    # Log results after we're done all episodes.
    log_results(filename, path_log, loss_log,m)            

def log_results(filename, path_log, loss_log,m):
    # Save the results to a file so we can graph it later.
    with open('results/logs/path_data-' + filename + '-' + str(m) + '-simple.csv', 'w') as pf:
        # path_length = list(map(lambda x:[x],path_log)) 
        # wr = csv.writer(pf)
        # for l in path_log:
        #     wr.writerows(l)
        wr = csv.writer(pf)
        wr.writerows(path_log)

    with open('results/logs/loss_data-' + filename + '-' + str(m) + '-simple.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)

# The features are [state,encoded action]
def get_features(state, action):
    # encode the action into a 3 element vector [1,0,0],[0,1,0], or[0,0,1]
    action_enc = np.zeros(3)
    action_enc[action] = 1
    features = np.hstack((state, action_enc))
    features = features.reshape((1,NUM_INPUT))
    return features

def process_minibatch(minibatch, model,batchSize):
    features_batch = []
    target_batch = []
    for mem in minibatch:
        state, action, reward, state_new = mem
        features = get_features(state,action)
        features_batch.append(features.reshape(NUM_INPUT))

        Q = np.zeros(3)
        for a in range(3):
            features=get_features(state_new,a)
            Q[a] = model.predict(features, batch_size=batchSize)
        maxQ = np.max(Q)

        # Check for terminal state and get predicted Q value
        if reward < 8000:  # non-terminal state
            target = (reward + (GAMMA * maxQ))
        else:  # terminal state
            target = reward

        target_batch.append(target)

    features_batch = np.array(features_batch)
    target_batch = np.array(target_batch)
    return features_batch, target_batch

def params_to_filename(params):
    return str(params['nn'][0]) + '-' + str(params['nn'][1]) + '-' + \
            str(params['batchSize']) + '-' + str(params['buffer'])

def launch_learn(params):
    filename = params_to_filename(params)
    print("Trying %s" % filename)
    # Make sure we haven't run this one.
    if not os.path.isfile('results/logs/loss_data-' + filename + '-simple.csv'):
        # Create file so we don't double test when we run multiple
        # instances of the script at the same time.
        open('results/logs/loss_data-' + filename + '-simple.csv', 'a').close()
        print("Starting test.")
        # Train.
        model = neural_net(NUM_INPUT, params['nn'])
        train(model, params)
    else:
        print("Already tested.")

if __name__ == "__main__":
    if TUNING:
        param_list = []
        nn_params = [[128, 128], [256, 256],
                     [512, 512], [1000, 1000]]
        batchSizes = [40, 100]
        buffers = [10000, 20000]

        for nn_param in nn_params:
            for batchSize in batchSizes:
                for buffer in buffers:
                    params = {
                        "batchSize": batchSize,
                        "buffer": buffer,
                        "nn": nn_param
                    }
                    param_list.append(params)

        for param_set in param_list:
            launch_learn(param_set)
    else:
        nn_param = [256,256]
        params = {
            "batchSize": 100,
            "buffer": 10000,
            "nn": nn_param
        }
        model = neural_net(NUM_INPUT, nn_param)
        train(model, params)
        # plot_model(model, to_file='saved-models/model_nn_01.png')
        
