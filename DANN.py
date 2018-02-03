import sys
import time

import data_helper
from flip_gradient import flip_gradient
from utils import *


def build_model(n_features, n_classes, batch_size, shallow_domain_classifier=True, n_domains=2):
    X = tf.placeholder(tf.float32, [None, n_features], name='X')  # Input data
    Y_ind = tf.placeholder(tf.int32, [None], name='Y_ind')  # Class index
    D_ind = tf.placeholder(tf.int32, [None], name='D_ind')  # Domain index
    train = tf.placeholder(tf.bool, [], name='train')  # Switch for routing data to class predictor
    l = tf.placeholder(tf.float32, [], name='l')  # Gradient reversal scaler

    Y = tf.one_hot(Y_ind, n_classes)  # convert number of classes to one hot
    D = tf.one_hot(D_ind, n_domains)  # convert number of domains to one hot

    # Feature extractor - single layer
    with tf.variable_scope('feature_extractor'):
        W0 = weight_variable([n_features, n_features * 2])
        b0 = bias_variable([n_features * 2])
        F = tf.nn.relu(tf.matmul(X, W0) + b0, name='feature')

    with tf.variable_scope('label_predictor'):
        f = tf.cond(train, lambda: tf.slice(F, [0, 0], [int(batch_size / 2), -1]), lambda: F)
        y = tf.cond(train, lambda: tf.slice(Y, [0, 0], [int(batch_size / 2), -1]), lambda: Y)

        W1 = weight_variable([n_features * 2, n_classes])
        b1 = bias_variable([n_classes])
        p_logit = tf.matmul(f, W1) + b1
        p = tf.nn.softmax(p_logit)
        p_loss = tf.nn.softmax_cross_entropy_with_logits(logits=p_logit, labels=y)

    with tf.variable_scope('domain_predictor'):
        # Domain predictor - shallow
        f_ = flip_gradient(F, l)

        if shallow_domain_classifier:
            W2 = weight_variable([n_features * 2, n_domains])
            b2 = bias_variable([n_domains])
            d_logit = tf.matmul(f_, W2) + b2
            d = tf.nn.softmax(d_logit)
            d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logit, labels=D)

        else:
            W2 = weight_variable([n_features * 2, n_features * 2])
            b2 = bias_variable([n_features * 2])
            h2 = tf.nn.relu(tf.matmul(f_, W2) + b2)

            W3 = weight_variable([n_features * 2, n_domains])
            b3 = bias_variable([n_domains])
            d_logit = tf.matmul(h2, W3) + b3
            d = tf.nn.softmax(d_logit)
            d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logit, labels=D)

    # Optimization
    pred_loss = tf.reduce_sum(p_loss, name='pred_loss')
    domain_loss = tf.reduce_sum(d_loss, name='domain_loss')
    total_loss = tf.add(pred_loss, domain_loss, name='total_loss')

    pred_train_op = tf.train.AdamOptimizer(0.01).minimize(pred_loss, name='pred_train_op')
    domain_train_op = tf.train.AdamOptimizer(0.01).minimize(domain_loss, name='domain_train_op')
    dann_train_op = tf.train.AdamOptimizer(0.01).minimize(total_loss, name='dann_train_op')

    # Evaluation
    p_acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y, 1), tf.arg_max(p, 1)), tf.float32), name='p_acc')
    d_acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(D, 1), tf.arg_max(d, 1)), tf.float32), name='d_acc')


def train_and_evaluate(op, X_src, y_src, X_tgt, y_tgt, grad_scale=None, batch_size=100, num_batches=2000, verbose=True):
    # Create batch builders
    g = tf.Graph()
    n_features = X_src.shape[1]
    n_classes = len(np.unique(y_src))

    with g.as_default():
        if op == 'Deep Domain Adaptation':
            train_op_name = 'dann_train_op'
            train_loss_name = 'total_loss'
            build_model(n_features=n_features, n_classes=n_classes, batch_size=batch_size,
                        shallow_domain_classifier=False)
        elif op == 'Domain Adaptation':
            train_op_name = 'dann_train_op'
            train_loss_name = 'total_loss'
            build_model(n_features=n_features, n_classes=n_classes, batch_size=batch_size)
        elif op == 'Domain Classification':
            train_op_name = 'domain_train_op'
            train_loss_name = 'domain_loss'
            build_model(n_features=n_features, n_classes=n_classes, batch_size=batch_size)
        elif op == 'Label Classification':
            train_op_name = 'pred_train_op'
            train_loss_name = 'pred_loss'
            build_model(n_features=n_features, n_classes=n_classes, batch_size=batch_size)
        else:
            raise ValueError('Invalid operation. Valid ops are: Deep Domain Adaptation, Domain Adaptation,'
                             ' Domain Classification, Label Classification')

        sess = tf.Session(graph=g)
        t = time.process_time()
        S_batches = batch_generator([X_src, y_src], batch_size // 2)
        T_batches = batch_generator([X_tgt, y_tgt], batch_size // 2)

        # Get output tensors and train op
        d_acc = sess.graph.get_tensor_by_name('d_acc:0')
        p_acc = sess.graph.get_tensor_by_name('p_acc:0')
        train_loss = sess.graph.get_tensor_by_name(train_loss_name + ':0')
        train_op = sess.graph.get_operation_by_name(train_op_name)

        sess.run(tf.global_variables_initializer())
        for i in range(num_batches):

            # If no grad_scale, use a schedule
            if grad_scale is None:
                p = float(i) / num_batches
                lp = 2. / (1. + np.exp(-10. * p)) - 1
            else:
                lp = grad_scale

            X0, y0 = S_batches.__next__()
            X1, y1 = T_batches.__next__()
            Xb = np.vstack([X0, X1])
            yb = np.hstack([y0, y1])
            D_labels = np.hstack([np.zeros(batch_size // 2, dtype=np.int32),
                                  np.ones(batch_size // 2, dtype=np.int32)])

            _, loss, da, pa = sess.run([train_op, train_loss, d_acc, p_acc],
                                       feed_dict={'X:0': Xb, 'Y_ind:0': yb, 'D_ind:0': D_labels,
                                                  'train:0': True, 'l:0': lp})

            if verbose and i % (num_batches // 20) == 0:
                print('loss: %f, domain accuracy: %f, class accuracy: %f' % (loss, da, pa))

        # Get final accuracies on whole dataset
        das, pas = sess.run([d_acc, p_acc], feed_dict={'X:0': X_src, 'Y_ind:0': y_src,
                                                       'D_ind:0': np.zeros(X_src.shape[0], dtype=np.int32),
                                                       'train:0': False,
                                                       'l:0': 1.0})
        dat, pat = sess.run([d_acc, p_acc], feed_dict={'X:0': X_tgt, 'Y_ind:0': y_tgt,
                                                       'D_ind:0': np.ones(X_tgt.shape[0], dtype=np.int32),
                                                       'train:0': False,
                                                       'l:0': 1.0})

        print('\n********' + str(op) + '********')
        print('Runtime: ', time.process_time() - t)
        print('Source domain: ', das)
        print('Source class: ', pas)
        print('Target domain: ', dat)
        print('Target class: ', pat)
        print('**********************************\n')


def main():
    if len(sys.argv) == 1:
        Xs, ys = data_helper.get_data('supernova-src')
        Xt, yt = data_helper.get_data('supernova-tgt')
    else:
        Xs, ys = data_helper.get_data(sys.argv[1])
        Xt, yt = data_helper.get_data(sys.argv[2])

    train_and_evaluate(op='Domain Classification', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt, grad_scale=-1.0)
    train_and_evaluate(op='Label Classification', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt)
    train_and_evaluate(op='Domain Adaptation', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt)
    train_and_evaluate(op='Deep Domain Adaptation', X_src=Xs, y_src=ys, X_tgt=Xt, y_tgt=yt)


if __name__ == '__main__':
    main()
