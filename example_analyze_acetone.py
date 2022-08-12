import numpy as np
from example_train_acetone import prepare_data, load_acetone_data
from nnanalyzers import DiscontinuousNeuralNetworkAnalyzer
import matplotlib.pyplot as plt
from matplotlib import cm
import yaml
from yaml import Loader
import warnings

# -------------------- SCRIPT'S MAIN PARAMETERS ---------------------
random_state = 2022

LOAD_ANALYZER = True  # FALSE: analysis is performed from scratch (it can take some time depending on Npts_epseval)
SAVE_RESULTS = True

folderpath = 'acetone_example'

Npts_epseval = 501
norm_ord = np.inf
weighted = True

topeps_to_print = [1, 2, 3]

# -------------------------------------------------------------------

ignore_somewarnings = True

if __name__ == '__main__':

    X, Y, N, T, V, X_le = load_acetone_data()

    if ignore_somewarnings:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

    analyzer_name = f'analyzer_B835_{random_state}'

    model_name = f'modelB835_{random_state}'

    analyzer = DiscontinuousNeuralNetworkAnalyzer()
    analyzer._load_keras_model(f'{folderpath}/{model_name}.h5',
                               custom_objects=None,
                               compile=False,
                               options=None
                               )

    with open(f'{folderpath}/{analyzer_name}.yml', 'r') as file:
        analyzer_dict = yaml.load(file, Loader=Loader)

    domain_intervals = analyzer_dict['domain_intervals']

    _, _, _, _, _, _, _, _, _, stdscal = prepare_data(X=X, Y=Y, T=T, V=V, random_state=random_state)

    xx1 = np.linspace(domain_intervals[0, 0], domain_intervals[1, 0], Npts_epseval)
    xx2 = np.linspace(domain_intervals[0, 1], domain_intervals[1, 1], Npts_epseval)

    XX1, XX2 = np.meshgrid(xx1, xx2)

    XX = np.hstack([XX1.flatten().reshape(XX1.size, 1), XX2.flatten().reshape(XX2.size, 1)])

    h_xx1 = (domain_intervals[1, 0] - domain_intervals[0, 0]) / (Npts_epseval - 1)
    h_xx2 = (domain_intervals[1, 1] - domain_intervals[0, 1]) / (Npts_epseval - 1)

    extent = (domain_intervals[0, 0] - h_xx1 / 2,
              domain_intervals[1, 0] + h_xx1 / 2,
              domain_intervals[0, 1] - h_xx2 / 2,
              domain_intervals[1, 1] + h_xx2 / 2
              )

    if LOAD_ANALYZER:
        # I DON'T IMPORT THE Npts_eval USED FOR THE EPS-RANK PROCEDURE, I WANT TO USE THE VALUE DEFINED ABOVE
        # Npts_epseval = analyzer_dict['Npts_epseval']
        epsweights_sorted = analyzer_dict['epsweights_sorted']
        maxabs_reltol_bounds = analyzer_dict['maxabs_reltol_bounds']
        maxabs_reltol_domain = analyzer_dict['maxabs_reltol_domain']

        analyzer.set_epsweights_sorted(epsweights_sorted=epsweights_sorted)
        analyzer.set_maxabs_reltol_bounds(maxabs_reltol_bounds=maxabs_reltol_bounds)
        analyzer.set_maxabs_reltol_domain(maxabs_reltol_domain=maxabs_reltol_domain)
    else:
        analyzer.fit_eps_rank(stdscal.transform(XX), norm_ord=norm_ord, weighted=weighted,
                              null_model=False,
                              add_maxabs_reltol_bounds=True,
                              add_maxabs_reltol_domain=True
                              )

    FIGURES = []
    for topeps_ind in topeps_to_print:
        _, XX_membership, _, _, _ = analyzer.contregions(stdscal.transform(XX),
                                                         which_boundaries=None,
                                                         top_eps=topeps_ind
                                                         )
        XX_membership_mat = XX_membership.reshape(XX1.shape)

        FIGURES.append(plt.figure())
        plt.imshow(XX_membership_mat,
                   origin='lower',
                   extent=extent,
                   vmin=XX_membership.min(),
                   vmax=XX_membership.max(),
                   cmap=cm.gist_rainbow,
                   interpolation=None
                   )
        plt.plot(X_le[:, 0], X_le[:, 1], 'k*--', linewidth=0.75)
        plt.colorbar()
        plt.title(f'k = {topeps_ind}')

        if SAVE_RESULTS:
            plt.savefig(f'{folderpath}/{model_name}_k{topeps_ind}pts{Npts_epseval}.png')
            plt.close()

    if not SAVE_RESULTS:
        plt.show()















