"""
Once a model is learned, use this to test and play it.
"""

from GameClass import GameClass
import numpy as np
from nn import neural_net

NUM_INPUT = 10


def play(model):

    car_distance = 0
    game_object = GameClass(True,60)

    # Do nothing to get initial.
    _, state = game_object.second_step((2))

    # Move.
    while True:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(state.reshape(1, NUM_INPUT), batch_size=40)))

        # Take action.
        reward, state = game_object.second_step(action)
        
        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)

        if reward > 9000:
            break


if __name__ == "__main__":
    saved_model = 'saved-models/nn_model.h5'
    model = neural_net(NUM_INPUT, [20,40])
    model.load_weights(saved_model)
    play(model)
