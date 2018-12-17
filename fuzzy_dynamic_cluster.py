import numpy as np


# direct cluster
class FCM(object):
    def __init__(self, data):
        self.lambd = 0
        self.data = data
        self.cluster = []
        self.F_S = []

    def standard(self):
        data_min, data_max = np.min(self.data, axis=0), np.max(self.data, axis=0)
        num_samples, num_shapes = np.shape(self.data)
        for i in range(num_samples):
            self.data[i, :] = (self.data[i, :])/data_max
            for j in range(num_shapes):
                self.data[i, j] = round(float(self.data[i, j]), 2)

    def matrix_alike(self):
        num_samples, num_shapes = np.shape(self.data)
        data = self.data
        r = np.zeros((num_samples, num_samples))
        # using max min method
        for i in range(num_samples):
            for j in range(num_samples):
                r[i, j] = np.sum(self.min(data[i, :], data[j, :]))/np.sum(self.max(data[i, :], data[j, :]))
                r[i, j] = round(r[i, j], 2)
        return r

    def max(self, a, b):
        a_or_b = []
        for (i, j) in zip(a, b):
            if i > j:
                a_or_b.append(i)
            else:
                a_or_b.append(j)
        return a_or_b

    def min(self, a, b):
        a_and_b = []
        for (i, j) in zip(a, b):
            if i < j:
                a_and_b.append(i)
            else:
                a_and_b.append(j)
        return a_and_b

    def merge_alike_class(self, a):
        b = []
        for i in range(len(a)):
            temp = []
            sign = False
            for j in range(len(a[i])):
                if len(b) != 0:
                    for k in range(len(b)):
                        if a[i][j] in b[k]:
                            b[k].extend(a[i])
                            b[k] = list(np.unique(b[k]))
                            sign = True
                            break
                if sign:
                    break
                temp.append(a[i][j])
            if sign:
                continue
            b.append(temp)
        return b

    def remove_same_cluster(self):
        length = len(self.cluster)
        temp = self.cluster.copy()
        for i in range(length-1):
            if self.cluster[i]['result'] == self.cluster[i+1]['result']:
                index = 0
                while True:
                    if temp[index]['lambd'] == self.cluster[i+1]['lambd']:
                        break
                    else:
                        index = index+1
                temp.pop(index)
        self.cluster = temp

    def cluster_t(self, T, lam):
        answer = T >= lam
        num_i, num_j = answer.shape
        x_index, y_index = [], []
        for i in range(num_i):
            for j in range(num_j):
                if answer[i, j]:
                    x_index.append(i+1)
                    y_index.append(j+1)
        num = list(np.unique(x_index))
        result = []
        for i in num:
            temp = []
            for j, k in zip(x_index, y_index):
                if i == j:
                    temp.append(k)
            result.append(temp)

        result = self.merge_alike_class(result)  # merge alike class
        return result

    # start cluster
    def fcm(self):
        self.standard()  # data standardization
        r = self.matrix_alike()  # create fuzzy alike matrix
        lambd = np.unique(r)  # get confidence level lambda
        lambd_length = len(lambd)
        for i in range(lambd_length):
            temp = {}
            temp['lambd'] = round(lambd[lambd_length-i-1], 2)
            temp['result'] = self.cluster_t(r, lambd[lambd_length-i-1])
            self.cluster.append(temp)
        self.remove_same_cluster()
        print('The result of cluster is  ', self.cluster)
        self.select_lambda()
        best = self.F_S.index(min(self.F_S))+1  # use the F-S function to be the validate measure of lambda
        print('The best lambda is  ', self.cluster[best]['lambd'])
        print('The best result of cluster is  ', self.cluster[best]['result'])

    def data_mean(self, data, index):
        if len(index) == 1:
            return data
        else:
            return np.mean(data, axis=0)

    def select_lambda(self):
        total_mean = np.mean(self.data, axis=0)
        length = len(self.cluster)
        for option in range(1, length-1):
            F_S = 0
            temp = 0
            for i in self.cluster[option]['result']:
                i = [j-1 for j in i]  # fix list index
                vi = self.data_mean(self.data[i, :], i)
                temp = 0
                for j in i:
                    temp = temp + (np.sum(np.square(self.data[j, :] - vi)) - np.sum(np.square(vi - total_mean)))
            F_S = F_S + temp
            self.F_S.append(F_S)


def main():
    data = np.array([[80., 10., 6., 2.],
                      [50., 1., 6., 4.],
                      [90., 6., 4., 6.],
                      [40., 5., 7., 3.],
                      [10., 1., 2., 4.]])
    fcm = FCM(data)
    fcm.fcm()


if __name__ == '__main__':
    main()
