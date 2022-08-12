import tensorflow as tf
import numpy as np
from nnlayers import DiscontinuityDense
from utils import np_heaviside
import copy


class DiscontinuousNeuralNetworkAnalyzer:
    def __init__(self, model=None):

        self.__model = None

        self.__model_path = None

        self.__model_copy = None
        self.__model_0 = None

        self.__inputlayers_inds = None
        self.__disclayers_inds = None
        self.__disclayers_weights = None

        self.__backbone_model_discbehavior = None
        self.__model_discbehavior = None

        self.__eps_rank = None
        self.__epsweights = []
        self.__epsweights_sorted = None
        self.__n_eps = None
        self.__maxabs_reltol_bounds = None
        self.__maxabs_reltol_domain = None

        if model is not None:

            self.__model = model

            self.__inputlayers_inds = [i for i in range(len(model.layers))
                                       if type(model.layers[i]) is tf.keras.layers.InputLayer]

            self.__disclayers_inds = [i for i in range(len(model.layers))
                                      if type(model.layers[i]) is DiscontinuityDense]

            if len(self.__disclayers_inds) == 0:
                raise ValueError('The model is not a discontinuous NN!')

            self.__disclayers_weights = [copy.deepcopy(layer.get_weights()) for layer in model.layers
                                         if type(layer) is DiscontinuityDense]

            eps_globalind = 0
            for ii in range(len(self.__disclayers_inds)):
                for jj in range(len(self.__disclayers_weights[ii][-1])):
                    self.__epsweights.append({
                        'value': copy.deepcopy(self.__disclayers_weights[ii][-1][jj]),
                        'disclayer_ind': copy.deepcopy(self.__disclayers_inds[ii]),
                        'eps_ind': jj,
                        'eps_globalind': eps_globalind
                    }
                    )
                    eps_globalind += 1

            self.__n_eps = eps_globalind

            self.__model_copy = tf.keras.models.clone_model(self.__model)
            self.__model_copy.set_weights(self.__model.get_weights())

            self.__backbone_model_discbehavior = tf.keras.models.clone_model(self.__model)
            self.__backbone_model_discbehavior.set_weights(self.__model.get_weights())

            BEHAV_LAYERS = []
            for ii in range(len(self.__disclayers_inds)):
                layer_ind = self.__disclayers_inds[ii]

                configs_ii = self.__backbone_model_discbehavior.layers[layer_ind].get_config()
                configs_ii['name'] = configs_ii['name'] + '_behav'

                BEHAV_LAYERS.append(DiscontinuityDense(
                    **configs_ii
                    )(self.__backbone_model_discbehavior.layers[layer_ind]._inbound_nodes[0].input_tensors)
                )

            self.__model_discbehavior = tf.keras.models.Model(
                inputs=[ll.output for ll in self.__backbone_model_discbehavior._input_layers],
                outputs=BEHAV_LAYERS
            )

            for ii in range(len(self.__disclayers_inds)):
                behav_weights_ii = copy.deepcopy(self.__disclayers_weights[ii])
                behav_weights_ii[-1][:] = 0.
                self.__model_discbehavior._output_layers[ii].set_weights(behav_weights_ii)

            self.__model_0 = tf.keras.models.clone_model(self.__model)
            self.__model_0.set_weights(self.__model.get_weights())

            for ii in range(len(self.__disclayers_inds)):
                layer_ind = self.__disclayers_inds[ii]
                new_weights = copy.deepcopy(self.__disclayers_weights[ii])
                new_weights[-1][:] = 0.
                self.__model_0.layers[layer_ind].set_weights(new_weights)

    @property
    def get_disclayers_inds(self):
        return self.__disclayers_inds

    @property
    def get_disclayers_weights(self):
        return self.__disclayers_weights

    @property
    def get_epsweights(self):
        return self.__epsweights

    @property
    def get_epsweights_sorted(self):
        return self.__epsweights_sorted

    @property
    def get_eps_rank(self):
        return self.__eps_rank

    @property
    def get_maxabs_reltol_bounds(self):
        return self.__maxabs_reltol_bounds

    @property
    def get_maxabs_reltol_domain(self):
        return self.__maxabs_reltol_domain

    def set_epsweights_sorted(self, epsweights_sorted):
        self.__epsweights_sorted = epsweights_sorted

    def set_maxabs_reltol_bounds(self, maxabs_reltol_bounds):
        self.__maxabs_reltol_bounds = maxabs_reltol_bounds

    def set_maxabs_reltol_bounds_from_data(self, X_data):
        n_disclayers = len(self.__disclayers_inds)
        if n_disclayers > 1:
            disc_behavior = np.hstack(self.__model_discbehavior.predict(X_data))
        else:
            disc_behavior = self.__model_discbehavior.predict(X_data)

        self.__maxabs_reltol_bounds = np.abs(disc_behavior).max(axis=0)

    def set_maxabs_reltol_domain(self, maxabs_reltol_domain):
        self.__maxabs_reltol_domain = maxabs_reltol_domain

    def set_maxabs_reltol_domain_from_data(self, X_data):
        self.__maxabs_reltol_domain = (X_data.max(axis=0) - X_data.min(axis=0)).max()

    def _load_keras_model(self, filepath,
                          custom_objects=None,
                          compile=True,
                          options=None):
        """
        Call the function tf.keras.models.load_model
        (see https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model)
        :param filepath: path of the keras model to load
        :param custom_objects: custom objects to load the model.
            The DiscontinuityDense layers are included by default.
        :param compile: boolean value
        :param options: loading options (e.g., custom_objects)
        :return:
        """

        if custom_objects is None:
            custom_objects = {}

        # WE ADD:
        # - DiscontinuityDense
        # - Zeros (required for discontinuity jump default initialization)
        custom_objects.update({DiscontinuityDense.__name__: DiscontinuityDense,
                               tf.keras.initializers.Zeros.__name__: tf.keras.initializers.Zeros}
                              )

        model = tf.keras.models.load_model(filepath,
                                           custom_objects=custom_objects,
                                           compile=compile,
                                           options=options)

        self.__init__(model=model)

        self.__model_path = filepath

    def _evaluate_eps_rank(self, X_data, epsweight_inds_list, norm_ord=np.inf, weighted=True, null_model=False):

        eps_rank = []
        sum_value = [np.abs(ee['value']) for ee in self.__epsweights]
        sum_value = np.sum(sum_value)

        for epsweight_ind in epsweight_inds_list:

            # print(f'----- ***** START EVALUATING eps #{epsweight_ind}')

            value = self.__epsweights[epsweight_ind]['value']
            disclayer_ind = self.__epsweights[epsweight_ind]['disclayer_ind']
            eps_ind = self.__epsweights[epsweight_ind]['eps_ind']

            weights_new = copy.deepcopy(self.__disclayers_weights[self.__disclayers_inds.index(disclayer_ind)])

            if null_model:
                weights_new[-1][:] = 0.
                weights_new[-1][eps_ind] = value

                model_0_pred = self.__model_0.predict(X_data)

                self.__model_0.layers[disclayer_ind].set_weights(weights_new)

                norm_ind = np.linalg.norm(
                    (model_0_pred - self.__model_0.predict(X_data)),
                    ord=norm_ord
                )

                weights_new[-1][:] = 0.
                self.__model_0.layers[disclayer_ind].set_weights(weights_new)
            else:
                weights_new[-1][eps_ind] = 0.

                self.__model_copy.layers[disclayer_ind].set_weights(weights_new)

                # print(f'-------------- ***** COMPUTE NORM W.R.T. eps #{epsweight_ind}')
                norm_ind = np.linalg.norm(
                    (self.__model.predict(X_data) - self.__model_copy.predict(X_data)),
                    ord=norm_ord
                )
                # print('-------------- @@@@@ COMPUTATION FINISHED')

                self.__model_copy.layers[disclayer_ind].set_weights(
                    self.__disclayers_weights[self.__disclayers_inds.index(disclayer_ind)]
                )

            if weighted:
                eps_rank.append((norm_ind * np.abs(value)) / sum_value)
            else:
                eps_rank.append(norm_ind)

        return eps_rank

    def fit_eps_rank(self, X_data, norm_ord=np.inf, weighted=True, null_model=False,
                     add_maxabs_reltol_bounds=True,
                     add_maxabs_reltol_domain=True
                     ):
        self.__eps_rank = self._evaluate_eps_rank(X_data,
                                                  list(range(self.__n_eps)),
                                                  norm_ord=norm_ord,
                                                  weighted=weighted,
                                                  null_model=null_model
                                                  )
        self.__epsweights_sorted = [self.__epsweights[i] for i in np.argsort(self.__eps_rank)]

        if add_maxabs_reltol_bounds:
            self.set_maxabs_reltol_bounds_from_data(X_data)

        if add_maxabs_reltol_domain:
            self.set_maxabs_reltol_domain_from_data(X_data)

    def __evaluate_contregions(self, X, mask=None):
        """
        Evaluate the continuity region vectors of the points in X.
        :param X: N-by-n array of N points in R^n
        :param mask: True/False array with same length of the tot. num. of disc. parameters (i.e., the epsilons)
        :return X_region_matrix: N-by-m matrix (m is the num. of disc. param.s selected by the mask) of continuity
            region vectors w.r.t. X
        """

        n_disclayers = len(self.__disclayers_inds)

        if n_disclayers > 1:
            X_region_matrix = np_heaviside(np.hstack(self.__model_discbehavior.predict(X)))
        else:
            X_region_matrix = np_heaviside(self.__model_discbehavior.predict(X))

        if mask is not None:
            X_region_matrix = X_region_matrix[:, mask]

        return X_region_matrix

    def __convert_to_mask(self, which_boundaries=None, top_eps=None):

        if which_boundaries is None:
            which_boundaries = [self.__epsweights_sorted[i]['eps_globalind']
                                for i in range(self.__n_eps)
                                ]
        else:
            which_boundaries = [self.__epsweights_sorted[i]['eps_globalind']
                                for i in range(self.__n_eps)
                                if self.__epsweights_sorted[i]['eps_globalind'] in which_boundaries
                                ]

        if top_eps is not None:
            which_boundaries = which_boundaries[-min(len(which_boundaries), top_eps):]

        mask = np.zeros(self.__n_eps).astype(np.bool)
        mask[which_boundaries] = True

        return mask, which_boundaries

    def contregions(self, X, which_boundaries=None, top_eps=None):

        mask, which_boundaries = self.__convert_to_mask(which_boundaries=which_boundaries, top_eps=top_eps)

        X_region_matrix = self.__evaluate_contregions(X, mask=mask)

        region_vectors, X_membership, region_vectors_counts = np.unique(X_region_matrix,
                                                                        axis=0,
                                                                        return_inverse=True,
                                                                        return_counts=True)

        return region_vectors, X_membership, region_vectors_counts, X_region_matrix, which_boundaries




