import tensorflow as tf
import numpy as np
import pandas as pd
from nnlayers import DiscontinuityDense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import yaml
from yaml import Loader

# -------------------- SCRIPT'S MAIN PARAMETERS ---------------------
random_state = 2022

SAVE_RESULTS = False

Nplot = 401

ACTIVATION = 'elu'
N_UNITS = 128
d_UNITS = 8

VERBOSE_TRAINING = True

# -------------------------------------------------------------------

train_configs = {
    'batch_size': 64,
    'epochs': 5000,
    'verbose': VERBOSE_TRAINING,
    'callbacks': [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=250,
            verbose=VERBOSE_TRAINING,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.75,
            patience=50,
            verbose=VERBOSE_TRAINING,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ],
    'shuffle': False,
    'class_weight': None,
    'sample_weight': None,
    'initial_epoch': 0,
    'steps_per_epoch': None,
    'validation_steps': None,
    'max_queue_size': 10,
    'workers': 1,
    'use_multiprocessing': False
}

OPTIMIZER = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam'
)

LOSS = tf.keras.losses.MeanSquaredError(
    name='mse_default'
)

METRICS = [
    tf.keras.losses.MeanAbsoluteError(name='mae_default')
]


def load_acetone_data(folder_path='acetone_example'):
    # ------- LOADING MAIN DATA ----------------
    acetone_df = pd.read_csv(f'{folder_path}/Acetone_DB_manualtri.csv', index_col='Unnamed: 0')
    # -------------------------------------

    # LOADING A MINMAX SCALER FIT FOR ACETONE DATA
    with open(f'{folder_path}/mMscaler_manualtri.yml', 'r') as file:
        mMscal_dict = yaml.load(file, Loader=Loader)

    mMscal = MinMaxScaler(mMscal_dict['params'])
    ks = [k for k in mMscal_dict.keys() if k != 'params']
    for k in ks:
        setattr(mMscal, k, mMscal_dict[k])

    # ------- LOAD AND PREPARE EXTRA DATA FOR PLOTS ----------------------
    df_original = pd.read_csv(f'{folder_path}/Acetonetable.csv')
    df0 = df_original.loc[:, ['State', 'K', 'MPa', 'g/l=kg/m3']]
    df = df0.loc[df0['MPa'] <= 10, :]
    df = df.rename(columns={'g/l=kg/m3': 'kg/m^3'})

    df_Le_vals = df.loc[df['State'] == 'Liquid at equilibrium', :]
    XY_le_base = df_Le_vals.loc[:, ['K', 'MPa', 'kg/m^3']].values

    XY_le = mMscal.transform(XY_le_base)
    X_le = XY_le[:, :2]
    # ----------------------------------------------------------------------

    X = acetone_df.values[:, :2]
    Y = acetone_df.values[:, -1]

    N = X.shape[0]
    T = 5600
    V = 1400

    return X, Y, N, T, V, X_le


def prepare_data(X, Y, T, V, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)

    N = X.shape[0]
    ii_perm = np.random.permutation(N)

    ii_train = ii_perm[:T]
    ii_val = ii_perm[T:T + V]
    ii_test = ii_perm[T + V:]

    X_train = X[ii_train, :]
    X_val = X[ii_val, :]
    X_test = X[ii_test, :]

    stdscal = StandardScaler()
    stdscal.fit(np.vstack([X_train, X_val]))

    Y_train = Y[ii_train]
    Y_val = Y[ii_val]
    Y_test = Y[ii_test]

    return ii_train, ii_val, ii_test, X_train, X_val, X_test, Y_train, Y_val, Y_test, stdscal


def create_new_B835(activation, n_units, d_units, random_state=None, name='no_name'):

    if random_state is not None:
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    ACTIVATION = activation
    N_UNITS = n_units
    d_UNITS = d_units

    I = tf.keras.layers.Input((2,), name='IL_0')
    D1 = tf.keras.layers.Dense(N_UNITS, activation=ACTIVATION, name='HL_1_Dense')(I)
    D2 = tf.keras.layers.Dense(N_UNITS, activation=ACTIVATION, name='HL_2_Dense')(D1)
    dD3 = DiscontinuityDense(d_UNITS, activation=ACTIVATION, name='HL_3_DiscontinuityDense')(D2)
    dD4 = DiscontinuityDense(d_UNITS, activation=ACTIVATION, name='HL_4_DiscontinuityDense')(dD3)
    D5 = tf.keras.layers.Dense(N_UNITS, activation=ACTIVATION, name='HL_5_Dense')(dD4)
    Dout = tf.keras.layers.Dense(1, activation='linear', name='OL_6_Dense')(D5)

    model = tf.keras.models.Model(inputs=I, outputs=Dout, name=name)
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    return model


def train_model(model, X_train, Y_train, X_val, Y_val, stdscal, train_configs=train_configs, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    train_configs['validation_data'] = (stdscal.transform(X_val), Y_val)

    history = model.fit(stdscal.transform(X_train), Y_train, **train_configs)

    return model, history


def make_plot(model, stdscal, X_test, Y_test, Nplot=401):
    h = 1 / (Nplot - 1)

    Xplot1, Xplot2 = np.meshgrid(np.linspace(0, 1, Nplot),
                                 np.linspace(0, 1, Nplot))

    Xplot = np.hstack([Xplot1.reshape(Xplot1.size, 1), Xplot2.reshape(Xplot2.size, 1)])

    Ypredplot = model.predict(stdscal.transform(Xplot)).flatten().reshape(Xplot1.shape)

    vmin = min(Y_test.min(), Ypredplot.min())
    vmax = max(Y_test.max(), Ypredplot.max())

    extent = (0 - h / 2, 1 + h / 2,
              0 - h / 2, 1 + h / 2
              )

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].scatter(X_test[:, 0], X_test[:, 1], c=Y_test, vmin=vmin, vmax=vmax)
    ax[0].set_title('test values')
    img1 = ax[1].imshow(Ypredplot,
                        origin='lower',
                        extent=extent,
                        vmin=vmin,
                        vmax=vmax
                        )
    ax[1].plot(X_le[:, 0], X_le[:, 1], 'k*--', linewidth=0.75)
    ax[1].set_title('predicted function')

    # Add an axes to the right of the main axes.1
    ax1_divider = make_axes_locatable(ax[1])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = fig.colorbar(img1, cax=cax1)

    fig_pred_actual = plt.figure()
    plt.imshow(Ypredplot,
               origin='lower',
               extent=extent,
               vmin=vmin,
               vmax=vmax
               )
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, edgecolors='black', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('predicted function and test values')

    fig_pred = plt.figure()
    plt.imshow(Ypredplot,
               origin='lower',
               extent=extent,
               vmin=vmin,
               vmax=vmax
               )
    plt.plot(X_le[:, 0], X_le[:, 1], 'k*--', linewidth=0.75)
    plt.colorbar()
    plt.title('predicted function')

    return fig, fig_pred_actual, fig_pred


if __name__ == '__main__':

    X, Y, N, T, V, X_le = load_acetone_data()

    (ii_train, ii_val, ii_test,
     X_train, X_val, X_test,
     Y_train, Y_val, Y_test,
     stdscal
     ) = prepare_data(X=X, Y=Y, T=T, V=V, random_state=random_state)

    model = create_new_B835(activation=ACTIVATION, n_units=N_UNITS, d_units=d_UNITS, random_state=random_state,
                            name=f'modelB835_{random_state}'
                            )

    model, history = train_model(model=model,
                                 X_train=X_train, Y_train=Y_train,
                                 X_val=X_val, Y_val=Y_val,
                                 stdscal=stdscal,
                                 train_configs=train_configs,
                                 random_state=random_state
                                 )

    Ypred_test = model.predict(stdscal.transform(X_test)).flatten()

    MAE_df = pd.DataFrame(np.abs(Y_test - Ypred_test).reshape(Y_test.size, 1), columns=['MAE'])

    fig, fig_pred_actual, fig_pred = make_plot(model=model, stdscal=stdscal,
                                               X_test=X_test, Y_test=Y_test, Nplot=Nplot)

    if SAVE_RESULTS:
        model.save(f'acetone_example/modelB835_{random_state}.h5')
        fig.savefig(f'acetone_example/modelB835_{random_state}_subplotcomp.png')
        fig_pred_actual.savefig(f'acetone_example/modelB835_{random_state}_comp.png')
        fig_pred.savefig(f'acetone_example/modelB835_{random_state}_pred.png')

        with open(f'acetone_example//modelB835_{random_state}_MAE.txt', 'w') as file:
            print(MAE_df.describe(percentiles=[0.03, 0.25, 0.50, 0.75, 0.97]), file=file)

        with open(f'acetone_example//modelB835_{random_state}_hist.yml', 'w') as file:
            yaml.dump(history.history, file)
    else:
        plt.show()





