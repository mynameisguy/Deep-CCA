import numpy as np
import scipy.io
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Dense, Flatten, Merge, merge
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import K, Model, Sequential, Input
import tensorflow as tf
import sys


batch_size = 37

def data():

    x_train = np.random.rand(1000, 2600)
    y_train = np.random.rand(1000, 2600)
    x_test = np.random.rand(100, 2600)
    y_test = np.random.rand(100, 2600)

    x_train = x_train.reshape(x_train.shape + (1,))
    y_train = y_train.reshape(y_train.shape + (1,))

    x_test = x_test.reshape(x_test.shape + (1,))
    y_test = y_test.reshape(y_test.shape + (1,))

    print(np.shape(x_train))
    print(np.shape(x_test))
    print(np.shape(y_train))
    print(np.shape(y_test))

    return x_train, y_train, x_test, y_test


def model_correlation_loss(representation_size, k_singular_values):
    global batch_size

    def keras_loss(y_true, y_pred):

        regularization_constant_1 = regularization_constant_2 = 1e-4
        epsilon = 1e-12

        o1 = o2 = int(y_pred.shape[1] // 2)

        h_1 = y_pred[:, 0:o1]
        h_2 = y_pred[:, o1:o1+o2]

        h_1 = tf.transpose(h_1)
        h_2 = tf.transpose(h_2)

        m = tf.shape(h_1)[1]

        centered_h_1 = h_1 - tf.cast(tf.divide(1, m),  tf.float32) * tf.matmul(h_1, tf.ones(shape=(m, m)))
        centered_h_2 = h_2 - tf.cast(tf.divide(1, m),  tf.float32) * tf.matmul(h_2, tf.ones(shape=(m, m)))

        sigma_hat_12 = tf.cast(tf.divide(1, m - 1),  tf.float32) * tf.matmul(centered_h_1, tf.transpose(centered_h_2))
        sigma_hat_11 = tf.cast(tf.divide(1, m - 1),  tf.float32) * tf.matmul(centered_h_1, tf.transpose(centered_h_1)) + regularization_constant_1 * tf.eye(num_rows=o1)
        sigma_hat_22 = tf.cast(tf.divide(1, m - 1),  tf.float32) * tf.matmul(centered_h_2, tf.transpose(centered_h_2)) + regularization_constant_2 * tf.eye(num_rows=o2)

        w_1, v_1 = tf.self_adjoint_eig(sigma_hat_11)
        w_2, v_2 = tf.self_adjoint_eig(sigma_hat_22)

        idx_pos_entries_1 = tf.where(tf.equal(tf.greater(w_1, epsilon), True))
        idx_pos_entries_1 = tf.reshape(idx_pos_entries_1, [-1, tf.shape(idx_pos_entries_1)[0]])[0]

        w_1 = tf.gather(w_1, idx_pos_entries_1)
        v_1 = tf.gather(v_1, idx_pos_entries_1)

        idx_pos_entries_2 = tf.where(tf.equal(tf.greater(w_2, epsilon), True))
        idx_pos_entries_2 = tf.reshape(idx_pos_entries_2, [-1, tf.shape(idx_pos_entries_2)[0]])[0]
        w_2 = tf.gather(w_2, idx_pos_entries_2)
        v_2 = tf.gather(v_2, idx_pos_entries_2)

        sigma_hat_rootinvert_11 = tf.matmul(tf.matmul(v_1, tf.diag(tf.divide(1,tf.sqrt(w_1)))), tf.transpose(v_1))
        sigma_hat_rootinvert_22 = tf.matmul(tf.matmul(v_2, tf.diag(tf.divide(1,tf.sqrt(w_2)))), tf.transpose(v_2))

        t_matrix = tf.matmul(tf.matmul(sigma_hat_rootinvert_11, sigma_hat_12), sigma_hat_rootinvert_22)

        if k_singular_values == representation_size:    # use all
            correlation = tf.sqrt(tf.trace(tf.matmul(tf.transpose(t_matrix), t_matrix)))
        else:
            w, v = tf.self_adjoint_eig(K.dot(K.transpose(t_matrix), t_matrix))
            non_critical_indexes = tf.where(tf.equal(tf.greater(w, epsilon), True))
            non_critical_indexes = tf.reshape(non_critical_indexes, [-1, tf.shape(non_critical_indexes)[0]])[0]
            w = tf.gather(w, non_critical_indexes)
            w = tf.gather(w, tf.nn.top_k(w[:, 2]).indices)
            correlation = tf.reduce_sum(tf.sqrt(w[0:representation_size]))

        return -correlation

    return keras_loss


def create_cnn_network(representation_size):

    data_dim = 1
    timesteps = 2600

    model = Sequential()
    model.add(Conv1D(10, 5, activation='relu', input_shape=(timesteps, data_dim)))
    model.add(BatchNormalization())
    model.add(Conv1D(10, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25, seed=6789))
    model.add(Conv1D(10, 5, activation='relu', ))
    model.add(BatchNormalization())
    model.add(Conv1D(10, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25, seed=6789))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25, seed=434))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25, seed=434))

    model.add(Dense(representation_size, activation='softmax'))

    return model


def create_model():

    # Create two identical neural networks for both representations
    representation_size = 50
    view_a = create_cnn_network(representation_size)
    view_b = create_cnn_network(representation_size)

    merged_model = Sequential()
    merged_model.add(Merge([view_a, view_b], mode='concat'))

    optimizer = Adam(lr=0.001)
    correlation_loss = model_correlation_loss(representation_size, k_singular_values=representation_size)
    merged_model.compile(loss=correlation_loss, optimizer=optimizer)

    return merged_model


def train(model, x_train, y_train, x_test, y_test, batch_size, n_epochs):

    x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

    # Labels all set to zero since they are not used in the cross-decomposition model
    """
    Reintroduce validation
    """
    print('Beginning training')
    model.fit(x=[x_train_, y_train_], y=np.zeros(len(x_train_)), validation_data=([x_val, y_val], np.zeros(len(x_val))), batch_size=batch_size, epochs=n_epochs, shuffle=True, verbose=1)

    result_val = model.evaluate([x_val, y_val], batch_size=batch_size, verbose=0)
    result_test = model.evaluate([x_test, y_test], batch_size=batch_size, verbose=0)

    print('Loss validation: {} Loss test: {}'.format(result_val, result_test))

    return model

def main():
    global batch_size
    batch_size = 37
    n_epochs = 5

    x_train, y_train, x_test, y_test = data()

    model = create_model()
    model.summary()

    # Train
    trained_model = train(model, x_train, y_train, x_test, y_test, batch_size, n_epochs)

if __name__ == '__main__':
    main()




