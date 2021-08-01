class DecisionTree:

    def __init__(self, logger, model, decision_tree_util):
        self.logger = logger
        self.model = model
        self.decision_tree_util = decision_tree_util

    def resize_data(self, environment, shape):
        x_data = self.model.get_x(environment).reshape(shape[0], shape[1]*shape[2])
        return x_data

    def train_model(self):
        labels_dict = {}
        aux = 0
        for string_y in np.unique(self.model.get_y(Environment.TRAIN)):
            labels_dict[string_y] = aux
            aux += 1
        x_train = self.resize_data(Environment.TRAIN, self.model.get_x(Environment.TRAIN).shape)
        x_test = self.resize_data(Environment.TEST, self.model.get_x(Environment.TEST).shape)
        y_train = self.convert_labels_to_numbers(labels_dict, self.model.get_y(Environment.TRAIN))
        y_test = self.convert_labels_to_numbers(labels_dict, self.model.get_y(Environment.TEST))

        xgboost_model = XGBClassifier()
        xgboost_model.fit(x_train, y_train, verbose=True, eval_set=[(x_test, y_test)])
        pickle.dump(xgboost_model, open("pima.pickle.dat", "wb"))
        """
        plt = plot_tree(xgboost_model, num_trees=1)
        fig = plt.gcf()
        fig.set_size_inches(30, 15)
        """
