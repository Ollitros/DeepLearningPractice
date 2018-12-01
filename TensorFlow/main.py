import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

my_data = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
]

dataset = tf.data.Dataset.from_tensor_slices(my_data)
print(dataset.output_shapes)
batches = dataset.batch(2)
print(batches.output_shapes)
iterator = batches.make_one_shot_iterator()
next_item = iterator.get_next()
print(next_item)    # [0, 1]
next_item = iterator.get_next()
print(next_item)    # [2, 3]
