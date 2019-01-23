import tensorflow as tf


# Creates a graph.
with tf.device('/device:GPU:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)


###
# python -m tensorboard.main --logdir=[PATH_TO_LOGDIR]
# [PATH_TO_LOGDIR] - without ' or ", + add '/'
# FOR EXAMPLE: "python -m tensorboard.main --logdir=logs/"
# where 'logs/' - is your directory where your log file placed from writer or smth else
# ERRORS: 1) tensorboard shows no graph -> try to insert writer code somewhere else near the session part
#            or after initialization of all variables
###

writer = tf.summary.FileWriter('logs')
writer.add_graph(tf.get_default_graph())

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
variable = sess.run([c])