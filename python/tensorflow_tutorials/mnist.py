# /python/tensorflow_tutorials/mnist_linear_softmax.py
#
# A simple exaple of using a single linear layer and softmax +
# one-hot encoded vectors to do softmax.
#
# See /LICENCE.md for Copyright information

import argparse

import math

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from tqdm import tqdm


def construct_linear_model(input_size, output_size):
    """Construct a linear model taking input_size and returning output_size."""
    weights = tf.Variable(tf.zeros([input_size, output_size]), name="weights")
    biases = tf.Variable(tf.zeros([10]), name="biases")

    return (weights, biases)


def train_mnist_model(data,
                      weights,
                      biases,
                      learning_rate=0.01,
                      batch_size=100,
                      epochs=100):
    """Train mnist model over one-hot encoded mnist data."""
    # Here we have the 'data' object which has a 'train' and 'test'
    # property. Each of those properties can give us batches of input/output
    # tensor pairs. The input tensors are 784x1 (think of them as a long
    # column of numbers, each number corresponding to a pixel in the image
    # based on how bright that pixel is). The output tensor is 1x1, with range
    # 0-10 based on which "class" the output image is. We'll convert it into
    # a "one-hot" encoded tensor later, meaning that it becomes a column
    # of ten numbers, with a "1" in the index of the correct class and zeroes
    # elsewhere.
    #
    # Also note that training occurs in what are called "batches". This is
    # purely a performance optimization. There's quite a lot of overhead
    # involved in sending stuff to and from the GPU, so we want the GPU
    # to do as much work as possible during that time. The GPU is massively
    # parallel, so we can quite easily do that - think of the batch size
    # as adding a third "depth" dimension to the tensors. Our input tensors
    # are now 784x1x100 and our output tensors are now 1x1x100 (which we
    # turn into 1x10x100 by one-hot encoding them).
    training_set = data.train
    input_tensor = tf.placeholder(tf.float32, [None, 784], name="input")

    # Now at the end of the computation, we do something called a "softmax"
    # on the output - this causes the magnitude of the vector to drop to 1
    # by applying a nonlinearity (in this case, e^n / sum[j](1 + e[j]^n),
    # where n is the relevant component). The TensorFlow tutorial explains
    # this as softmax(x) = normalize(exp(x)), which is probably a nicer way to
    # think about it. It is the exponent of a component divided by the
    # exponents of every component.
    output_result = tf.nn.softmax(tf.matmul(input_tensor, weights) + biases)

    # To train the model we need a cost function. We can think of our
    # ten-dimensional output as a "probability" that the input is of that
    # class, since sum(output_result) is a probability density function as
    # the softmax guarantees that it will add up to 1.
    #
    # One good cost function when dealing with discrete probability density
    # functions is called "cross-entropy" loss. To understand this function,
    # think of what our output and one-hot encoded ground truth tensors
    # might look like:
    #
    # output_tensor: [ 0.01, 0.02, 0.05, 0.05, 0.65, 0.01, 0.01, 0.0, 0.01, 0.099 ]
    # ground_truth:  [ 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.0, 0.0, 0.0 ]
    #
    # Notice that the ground truth only needed to use one cell to represent
    # the result whereas the output_tensor used nine cells, and the "strongest"
    # was 0.65. Thus, the probability distribution doesn't match very well. We
    # want to "penalize" it for spreading out the probability distribution
    # too thin and not making the fifth cell higher. So we define a function
    # as follows:
    #    E(output_tensor, ground_truth) = -sum[i](ground_truth[i],
    #                                             log(output_tensor[i]))
    #
    # Why the negative? Because values in our output_tensor will never be
    # greater than 1, so log of a number between 1 and 0 is always negative
    # (as it would solve for a negative exponent). So we need to negate
    # the result to get a positive error.
    ground_truth = tf.placeholder(tf.float32, [None, 10], name="ground_truth")

    # Here tf.reduce_sum sums up all the elemnts in the second dimension,
    # since we pointwise multiplied everything together. tf.reduce_mean
    # computes the mean loss over all examples in the batch
    # (the first dimension).
    cross_entropy_loss = tf.reduce_mean(
        -tf.reduce_sum(ground_truth * tf.log(output_result),
                       reduction_indices=[1])
    )

    # Finally, we have a 'stepper' which is a gradient descent optimizer
    # taking the loss over all the examples in the entire training set,
    # computing all the derivatives of the weights with respect to the
    # loss and adjusting the weights so that the loss is minimized. 0.5
    # is the "learning rate", or a scalar that we multiply the gradients
    # by to ensure that we don't overshoot the optima.
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(
        cross_entropy_loss
    )

    # Measure accuracy: Is the highest-valued component of
    # output_tensor at the same index of the highest valued
    # component of ground truth tensor? 1 for accurate, in
    # that component, 0 otherwise.
    correct_prediction = tf.equal(tf.argmax(output_result, 1),
                                  tf.argmax(ground_truth, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    # Initialize all our global variables
    tf.global_variables_initializer().run()

    # Run the training loop
    for epoch in range(epochs):
        inputs_batch, ground_truth_batch = training_set.next_batch(
            batch_size
        )

        # We "run" the training step, cross_entropy_loss - which just
        # returns the value of the cross_entropy_loss function and
        # accuracy - which return the value of the accuracy.
        #
        # Note as well that we "one-hot" encode our ground_truth batch
        # since the training data only has the label class number,
        # not the label class vector
        yield sess.run([train_step, cross_entropy_loss, accuracy],
                       feed_dict={
                           input_tensor: inputs_batch,
                           ground_truth: tf.one_hot(ground_truth_batch, 10).eval()
                       })


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("""TensorFlow MNIST Linear Softmax.""")
    parser.add_argument("--epochs",
                        default=1000,
                        type=int,
                        required=False,
                        help="""Number of epochs to run training for.""")
    parser.add_argument("--learning-rate",
                        default=0.01,
                        type=float,
                        required=False,
                        help="""Learning rate to use.""")
    parser.add_argument("--batch-size",
                        default=100,
                        type=int,
                        required=False,
                        help="""Batch size to use.""")
    parser.add_argument("--data-directory",
                        default="mnist_data/",
                        type=str,
                        required=False,
                        help="""Where to store uncompressed data.""")
    result = parser.parse_args()

    data = input_data.read_data_sets(result.data_directory)

    # Lets create some variables in the computation graph. This is essentially
    # the process of fitting 10 784 dimensional lines (784 inputs, 10 outputs),
    # plus an offset for each of those lines.
    #
    # Well, mathematically that's what happens though thinking of 10 784
    # dimensional lines is insane so we resort to our good old fashioned
    # neural net diagram where we show how each of the 784 dimensions of the
    # inputs are connected to the output by lines, where each of those lines
    # is the weight given to that dimension in the output for the corresponding
    # output dimension.
    #
    # Basically y[10] = x[784] * W[784, 10] + B[10]
    weights, biases = construct_linear_model(784, 10)

    with tqdm(train_mnist_model(data,
                                weights,
                                biases,
                                epochs=result.epochs,
                                learning_rate=result.learning_rate,
                                batch_size=result.batch_size),
              total=result.epochs) as bar:
        for _, loss, acc in bar:
            bar.set_description("Training loss: {}, acc {}".format(
                math.trunc(loss * 100) / 100,
                math.trunc(acc * 100) / 100
            ))
