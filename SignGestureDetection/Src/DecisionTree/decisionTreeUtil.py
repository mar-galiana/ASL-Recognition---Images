class DecisionTreeUtil:

    def __init__(self, model):
        self.model = model

    @staticmethod
    def convert_labels_to_numbers(labels_dict, labels):
        for aux in range(len(labels)):
            labels[aux] = labels_dict.get(labels[aux])

        return labels
