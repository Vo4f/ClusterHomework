import numpy as np
import calculator
import loader
import utils


def calc_center(item, centorid):
    min_dist = 10000000
    min_res = None
    for n, i in enumerate(centorid):
        dist = calculator.euclidean_dist(i, item)
        if dist < min_dist:
            min_dist = dist
            min_res = n
    return min_res


def calc_new_center(term):
    return np.mean(term, axis=0)


class Kmeans(object):
    def __init__(self, k, file, max_iter=50, precision=5):
        self.k = k
        self.file = file
        self.max_iter = max_iter
        self.precision = precision

    def cal_kmeans(self):
        raw_data = loader.data_load(self.file)
        centroid = utils.init_center(raw_data, self.k)
        converged = False
        iter_count = 0
        while not converged:
            iter_count += 1
            print('iter:' + str(iter_count))
            class_list = [[] for i in range(self.k)]
            old_centroid = [[round(j, self.precision) for j in i] for i in centroid]
            for item in raw_data:
                centroid_index = calc_center(item, centroid)
                class_list[centroid_index].append(item)
            for ind, term in enumerate(class_list):
                centroid[ind] = calc_new_center(term)
            cur_centroid = [[round(j, self.precision) for j in i] for i in centroid]
            if utils.isconverged(old_centroid[-1:], cur_centroid[-1:]) or iter_count > self.max_iter:
                converged = True
                utils.display_res(class_list)


if __name__ == '__main__':
    km = Kmeans(3, 'waveform012.data', 500, 10)
    km.cal_kmeans()