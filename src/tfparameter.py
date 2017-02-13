# -*- coding: utf-8 -*-

class TFRunParameter:
    # Parameters
    def __init__(self, learning_rate, training_iters, batch_size, display_step):
        # 0.01, 10000, 10, 10
        self.learning_rate = learning_rate
        self.training_iters = training_iters
        self.batch_size = batch_size
        self.display_step = display_step


class TFNNParameter:
    def __init__(self, n_input, n_steps, n_hidden, n_classes):
        # Network Parameters
        # 1, 20, 2, 5
        self.n_input = n_input  # data input
        self.n_steps = n_steps  # timesteps
        self.n_hidden = n_hidden # hidden layers
        self.n_classes = n_classes