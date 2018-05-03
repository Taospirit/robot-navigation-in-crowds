"""
Once a model is learned, use this to test and play it.
"""

from GameClass import GameClass
from trainning import get_features
import numpy as np
from nn import neural_net
from keras.models import load_model

FPS = 60

def play(model):

    path_length = 0
    gameObject = GameClass(draw_screen = True, display_path = True, fps = FPS)

    # Do nothing to get initial.
    _, state = gameObject.frame_step((2))

    # Move.
    while True:
        path_length += 1

        # Choose action.
        Q = np.zeros(3)
        for a in range(3):
            features = get_features(state,a)
            Q[a] = model.predict(features)
            action = (np.argmax(Q))

        # Take action.
        reward, state = gameObject.frame_step(action)
        
        # Tell us something.
        if path_length % 1000 == 0:
            print("Current distance: %d frames." % path_length)

        if reward > 8000:
            break


if __name__ == "__main__":
    m=6
    n=9
    p=9
    # reach goal(no path):
    # reach goal: 11, 12
    
    # saved_model = load_model('saved-models/model_nn-1000-1000-100-10000-' + str(m)+ '.h5')
    # saved_model = load_model('saved-models/model_nn-256-256-100-10000-' + str(m)+ '-noPath.h5')
    # saved_model = load_model('saved-models/model_nn-128-128-100-10000-' + str(n)+ '.h5')
    saved_model = load_model('saved-models/model_nn-512-512-100-10000-' + str(p)+ '.h5')

    play(saved_model)

