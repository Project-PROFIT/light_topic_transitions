"""    
    Authors: Victor Mireles  & Artem Revenko for Semantic Web Company
    Cite:
        Mireles V., Revenko A. "Evolution of Semantically Identified Topics"
        CEUR vol 1923 (2017)
        http://ceur-ws.org/Vol-1923/article-06.pdf
"""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import sklearn.decomposition.nmf as sknmf


def get_topics(X, n_topics):
    """
    X = W*H

    :param X: vectorized input
    :param n_topics: number of topics
    :return: W, H, nmf
    """
    nmf = sknmf.NMF(n_components=n_topics, l1_ratio=0.5, init='nndsvd')
    W = nmf.fit_transform(X)
    H = nmf.components_
    return W, H, nmf


def get_top_words(H, feature_names, n_top_words=7):
    out = []
    for topic in H:
        out.append([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]])
    return out


def get_topic_transition(H_t1, H_t):
    def func(x):
        M_ = np.array(x).reshape(H_t.shape[0], H_t1.shape[0])
        return np.linalg.norm(H_t - np.dot(M_, H_t1))

    M1 = H_t.dot(np.linalg.pinv(H_t1))  # initial value
    x_0 = M1.reshape(1, M1.shape[0] * M1.shape[1])
    x_bounds = [(0, 1)] * x_0.shape[1]
    x, f, d = fmin_l_bfgs_b(func, x_0, bounds=x_bounds, approx_grad=True)
    M = np.asarray(x).reshape(M1.shape)

    return M


def draw_topic_transition(M, x_ticks, y_ticks, show=False):
    import plotly.offline as po
    import plotly.graph_objs as go

    data = [
        go.Heatmap(
            z=M,
            x=x_ticks,
            y=y_ticks,
        )
    ]
    layout = go.Layout(
        title='Topic transition',
        xaxis=dict(
            title='Current timeframe',
            ticks='',
            showticklabels=False,
            showgrid=False,
        ),
        yaxis = dict(
            title='Previous timeframe',
            ticks='',
            showticklabels=False,
            showgrid=False,
        ),
    )
    fig = go.Figure(data=data, layout=layout)

    if show:
        out_type = 'file'
    else:
        out_type = 'div'
    plot_div = po.plot(fig,
                       filename='simple-colorscales-colorscale.html',
                       auto_open=show,
                       output_type=out_type)
    return plot_div


def draw_rec_err(l1ratio, location, X):
    import matplotlib.pyplot as plt
    x = []
    y = []
    z = []
    for n_topics in range(10, 80, 2):
        nmf = sknmf.NMF(n_components=n_topics, l1_ratio=l1ratio, init='nndsvd')
        W = nmf.fit_transform(X)
        print('\nNumber of topics: {}, reconstruction error: {}'.format(
            n_topics, nmf.reconstruction_err_
        ))
        print('Topics: shape: {}, nonzeros: {}, density: {}'.format(
            nmf.components_.shape, len(nmf.components_.nonzero()[0]),
            len(nmf.components_.nonzero()[0]) / (
                nmf.components_.shape[0] * nmf.components_.shape[1])
        ))
        x.append(n_topics)
        y.append(nmf.reconstruction_err_)
        z.append(len(nmf.components_.nonzero()[0]) /
                 (nmf.components_.shape[0] * nmf.components_.shape[1]))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(x, y, 'bo', x, y, 'k')
    plt.title('Reconstruction error')
    plt.grid(True)
    plt.ylabel('Reconstruction error')
    plt.xlabel('Number of Topics')

    plt.subplot(212)
    plt.plot(x, z, 'ro', x, z, 'k')
    plt.title('Density')
    plt.grid(True)
    plt.ylabel('Density')
    plt.xlabel('Number of Topics')
    plt.tight_layout()
    plt.savefig(location)
    plt.close()


if __name__ == '__main__':
    pass
