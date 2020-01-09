import numpy as np
import tensorflow as tf

from modeling import attention_layer

batch_size = 2
num_attention_heads = 2
size_per_head = 3
seq_length = 4

query_kernel = np.array([
    [-0.07848196, -0.18097023, 0.06933199, -0.07760319, 0.11389876, 0.05236414],
    [-0.02015782, 0.00233333, -0.00281469, -0.01525305, 0.17362033, -0.01600084],
    [0.00521428, 0.06063714, -0.10533229, 0.0228875, -0.00108843, -0.05974746],
    [-0.05530503, 0.06056419, 0.099603, 0.04929306, 0.08636444, 0.08424559],
    [0.02739674, -0.08676406, -0.0819858, 0.03834791, -0.03903558, 0.01903536],
    [0.01325864, 0.07587593, 0.20709228, -0.0421985, -0.10500058, -0.08004139]
], dtype=np.float32)
query_bias = np.array([-0.01566293, -0.01429354, -0.02946532, 0.02332242, -0.03551506, 0.00519018], dtype=np.float32)

key_kernel = np.array([
    [-0.19046976, -0.052291, 0.00774184, -0.04793982, -0.03272828, -0.07022775],
    [0.05397043, 0.22157724, -0.28796428, -0.13628182, 0.10769557, -0.04396444],
    [0.11023977, 0.11277004, -0.17019109, -0.00998783, -0.13503011, 0.03862515],
    [-0.00570178, -0.03683843, -0.09878516, -0.08536254, -0.20706373, 0.07736684],
    [0.09753255, 0.08549864, 0.07573727, -0.08459285, 0.11262332, -0.06660723],
    [-0.05978908, 0.04687774, 0.20048976, -0.15552515, -0.09287686, -0.05736409]
], dtype=np.float32)
key_bias = np.array([0.01119683, -0.00749641, 0.00929781, -0.00789247, 0.00374282, -0.0203852], dtype=np.float32)

value_kernel = np.array([
    [0.18298741, 0.13052676, 0.13003705, -0.07762788, -0.11298412, -0.09672086],
    [-0.27567647, -0.11159269, -0.20191047, -0.04961415, 0.03338585, -0.00217377],
    [0.0080993, -0.0092568, -0.07923323, -0.09595821, -0.0724212, 0.00234286],
    [0.08350474, 0.10685625, -0.03265393, 0.12026393, 0.11865459, 0.03879681],
    [0.09247954, -0.08354547, -0.04044447, 0.05576184, 0.063286, -0.06426957],
    [0.11189654, 0.04743394, 0.04952021, 0.06824017, -0.0718908, 0.06118326]
], dtype=np.float32)
value_bias = np.array([-0.01532887, -0.02567805, 0.02993296, 0.00255634, 0.03075514, -0.02086536], dtype=np.float32)


if __name__ == '__main__':
    tensor = tf.placeholder(tf.float32, shape=[batch_size * seq_length, num_attention_heads * size_per_head])
    neg_attention_mask = tf.placeholder(tf.float32, shape=[batch_size, seq_length])

    tensor_ = np.arange(48, dtype=np.float32)
    tensor_ = np.reshape(tensor_, (batch_size*seq_length, num_attention_heads * size_per_head))

    neg_attention_mask_ = np.zeros((batch_size, seq_length), dtype=np.float32)
    neg_attention_mask_[0, 2] = 1
    neg_attention_mask_[0, 3] = 1
    neg_attention_mask_r = tf.reshape(neg_attention_mask, [batch_size, 1, 1, seq_length])
    neg_attention_mask_r *= tf.ones(shape=[batch_size, num_attention_heads, seq_length, seq_length],
                                  dtype=tf.float32)

    with tf.Session() as sess:
        context_layer = attention_layer(tensor, neg_attention_mask_r,
                                        query_kernel=query_kernel, query_bias=query_bias,
                                        key_kernel=key_kernel, key_bias=key_bias,
                                        value_kernel=value_kernel, value_bias=value_bias,
                                        num_attention_heads=num_attention_heads,
                                        size_per_head=size_per_head,
                                        batch_size=batch_size,
                                        seq_length=seq_length)

        context_layer_, attention_probs_ = sess.run([context_layer, attention_probs], feed_dict={
            tensor: tensor_,
            neg_attention_mask: neg_attention_mask_,
        })
        print(context_layer_)