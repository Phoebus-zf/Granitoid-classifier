import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def create_cube():
    # create a 3D dataset on a cube surface 
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    z = np.linspace(0, 1, 20)

    xyz = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    # # add a label to each point
    xyz = np.concatenate((xyz, np.zeros((xyz.shape[0], 1))), axis=1)
    # xyz[:2000, 3] = 4
    # xyz[2000:4000, 3] = 1
    # xyz[:, 4000:] = 0
    # # name the columns
    xyz = pd.DataFrame(xyz, columns=['x', 'y', 'z', 'label'])
    xy = xyz[xyz['z'] == 0]
    xz = xyz[xyz['y'] == 0]
    yz = xyz[xyz['x'] == 0]
    surface = pd.concat([xy, xz, yz])
    # add noise to the dataset
    surface['x'] += np.random.normal(-0.01, 0.01, surface.shape[0])
    surface['y'] += np.random.normal(-0.01, 0.01, surface.shape[0])
    surface['z'] += np.random.normal(-0.01, 0.01, surface.shape[0])

    # assign a label to each point

    # surface['label'] = 0
    # surface.loc[surface['x'] == 0, 'label'] = 1
    # surface.loc[surface['y'] == 0, 'label'] = 2
    # surface.loc[surface['z'] == 0, 'label'] = 3

    return surface


class Explain:
    def __init__(self, X, X_2d, columns=None, epsilon=0.1, color_nums=5, alpha=1):
        self.X = np.array(X)
        if type(X) == pd.DataFrame:
            self.columns = X.columns
        elif columns:
            self.columns = columns
        else:
            self.columns = range(X.shape[1])
        self.X_2d = X_2d
        self.epsilon = epsilon
        self.alpha = alpha
        self.color_nums = color_nums
        self.scores = self.get_relative_variance_matrix()
        
        
       

    # For a 2D projected point q, we define its 2D neighborhood ν2d and return the neiborhood's index
    def get_neighborhood(self, q, alph=1):
        distances = np.linalg.norm(self.X_2d - q, axis=1)
        radius = alph * self.epsilon * np.max(distances)
        indices = np.where(distances <= radius)[0]
        # print('indices: ', indices)
        if len(indices) == 1:
            print('1 indices error: ', indices, q)
        return indices

    # normalized relative variance ω of dimension j over the neighborhood μ
    def get_relative_variance(self, q, j):
        neighborhood = self.X[self.get_neighborhood(q)]
        lv = np.var(neighborhood[:, j])
        gv = np.var(self.X[:, j])
        # print('gv: ', gv)
        # print(np.var(self.X, axis=0))
        norm = np.sum(np.var(neighborhood, axis=0) / np.var(self.X, axis=0))
        return (lv / gv) / norm

    def get_relative_variance_matrix(self):
        n, d = self.X.shape
        relative_variance_matrix = np.zeros((n, d))
        for i in range(n):
            for j in range(d):
                relative_variance_matrix[i, j] = self.get_relative_variance(self.X_2d[i], j)

        relative_variance_matrix = pd.DataFrame(relative_variance_matrix, columns=self.columns)
        
        relative_variance_matrix['color'] = relative_variance_matrix.idxmin(axis=1)
        relative_variance_matrix['color_plot'] = relative_variance_matrix['color']
        if len(self.columns) > self.color_nums + 1:
            # select the top 8 colors 
            colors = relative_variance_matrix['color'].value_counts()[:self.color_nums]
            # print(colors)
            # only the top 8 frequet colors are considered. other colors are assigned to 'other'
            # relative_variance_matrix['color_plot'] = 'others'
            relative_variance_matrix['color_plot'] = relative_variance_matrix['color'].apply(lambda x: x if x in colors else 'others')   

        return relative_variance_matrix

    def get_confidence_score(self, row):
        index = row.name
        neighor_index = self.get_neighborhood(self.X_2d[index], self.alpha)
        print('@debug:', self.scores.loc[neighor_index, row['color']])
        up = np.sum(self.scores.loc[neighor_index, row['color']])
        print('up: ', up)
        down = np.sum(np.sum(self.scores.iloc[neighor_index, :-2]))
        print('down: ', down)
        return  up / down
      

    def get_confidence_scores(self):
        scores = self.scores
        scores['confidence'] = scores.apply(self.get_confidence_score, axis=1)
        scores['confidence_normalized'] =  1 - scores['confidence'] / scores['confidence'].max()
        return scores
