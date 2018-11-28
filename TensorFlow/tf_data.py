import tensorflow as tf


tf.enable_eager_execution()

# ***--------------------------------------------------------------------------------------------------------------*** #
# Dataset iteration
my_data = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
]

dataset = tf.data.Dataset.from_tensor_slices(my_data)
iterator = dataset.make_one_shot_iterator()
next_item = iterator.get_next()
print(next_item)    # [0, 1]
next_item = iterator.get_next()
print(next_item)    # [2, 3]

# ***--------------------------------------------------------------------------------------------------------------*** #
