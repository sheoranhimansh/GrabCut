import sys
import os
import numpy as np
import math
import cv2 as cv
import igraph as ig
from sklearn.cluster import KMeans

def score_formula(mult,mat):
    score = np.exp(-.5 * mult) / np.sqrt(2 * np.pi)/np.sqrt(np.linalg.det(mat))
    return score

class GaussianMixture:
    def __init__(self, X):
        self.n_components = 4
        self.n_features = X.shape[1]
        self.n_samples = np.zeros(self.n_components)

        self.coefs = np.zeros(self.n_components)
        self.means = np.zeros((self.n_components, self.n_features))
        self.covariances = np.zeros(
            (self.n_components, self.n_features, self.n_features))

        self.init_with_kmeans(X)

    def init_with_kmeans(self, X):
        label = KMeans(n_clusters=self.n_components, n_init=1).fit(X).labels_
        self.fit(X, label)

    def calc_score(self, X, ci):
        score = np.zeros(X.shape[0])
        if self.coefs[ci] > 0:
            diff = X - self.means[ci]
            Tdiff = diff.T
            inv_cov = np.linalg.inv(self.covariances[ci])
            dot = np.dot(inv_cov, Tdiff)
            Tdot = dot.T
            mult = np.einsum('ij,ij->i', diff, Tdot)
            score = score_formula(mult,self.covariances[ci])
        return score

    def calc_prob(self, X):
        prob = []
        for ci in range(self.n_components):
            score = np.zeros(X.shape[0])
            if self.coefs[ci] > 0:
                diff = X - self.means[ci]
                Tdiff = diff.T
                inv_cov = np.linalg.inv(self.covariances[ci])
                dot = np.dot(inv_cov, Tdiff)
                Tdot = dot.T
                mult = np.einsum('ij,ij->i', diff, Tdot)
                score = score_formula(mult,self.covariances[ci])
            prob.append(score)
        ans = np.dot(self.coefs, prob)
        return ans

    def which_component(self, X):
        prob = []
        for ci in range(self.n_components):
            score = self.calc_score(X,ci)
            prob.append(score)
        prob = np.array(prob).T
        return np.argmax(prob, axis=1)

    def fit(self, X, labels):
        assert self.n_features == X.shape[1]
        self.n_samples[:] = 0
        self.coefs[:] = 0
        uni_labels, count = np.unique(labels, return_counts=True)
        self.n_samples[uni_labels] = count
        variance = 0.01
        for ci in uni_labels:
            n = self.n_samples[ci]
            sum = np.sum(self.n_samples)
            self.coefs[ci] = n / sum
            self.means[ci] = np.mean(X[ci == labels], axis=0)
            if self.n_samples[ci] <= 1:
                self.covariances[ci] = 0
            else:
                self.covariances[ci] =  np.cov(X[ci == labels].T)
            det = np.linalg.det(self.covariances[ci])
            if det <= 0:
                self.covariances[ci] += np.eye(self.n_features) * variance
                det = np.linalg.det(self.covariances[ci])

def construct_gc_graph(img,mask,gc_source,gc_sink,fgd_gmm,bgd_gmm,gamma,rows,cols,left_V,upleft_V,up_V,upright_V):
    bgd_indexes = np.where(mask.reshape(-1) == DRAW_BG['val'])
    fgd_indexes = np.where(mask.reshape(-1) == DRAW_FG['val'])
    pr_indexes = np.where(np.logical_or(mask.reshape(-1) == DRAW_PR_BG['val'],mask.reshape(-1) == DRAW_PR_FG['val']))
    print('bgd count: %d, fgd count: %d, uncertain count: %d' % (len(bgd_indexes[0]), len(fgd_indexes[0]), len(pr_indexes[0])))
    edges = []
    gc_graph_capacity = []
    edges.extend(list(zip([gc_source] * pr_indexes[0].size, pr_indexes[0])))
    _D = -np.log(bgd_gmm.calc_prob(img.reshape(-1, 3)[pr_indexes]))
    gc_graph_capacity.extend(_D.tolist())
    edges.extend(list(zip([gc_sink] * pr_indexes[0].size, pr_indexes[0])))
    _D = -np.log(fgd_gmm.calc_prob(img.reshape(-1, 3)[pr_indexes]))
    gc_graph_capacity.extend(_D.tolist())
    edges.extend(list(zip([gc_source] * bgd_indexes[0].size, bgd_indexes[0])))
    gc_graph_capacity.extend([0] * bgd_indexes[0].size)
    edges.extend(list(zip([gc_sink] * bgd_indexes[0].size, bgd_indexes[0])))
    gc_graph_capacity.extend([9 * gamma] * bgd_indexes[0].size)
    edges.extend(list(zip([gc_source] * fgd_indexes[0].size, fgd_indexes[0])))
    gc_graph_capacity.extend([9 * gamma] * fgd_indexes[0].size)
    edges.extend(list(zip([gc_sink] * fgd_indexes[0].size, fgd_indexes[0])))
    gc_graph_capacity.extend([0] * fgd_indexes[0].size)

    img_indexes = np.arange(rows*cols,dtype=np.uint32).reshape(rows,cols)
    temp1 = img_indexes[:, 1:]
    temp2 = img_indexes[:, :-1]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc_graph_capacity.extend(left_V.reshape(-1).tolist())
    temp1 = img_indexes[1:, 1:]
    temp2 = img_indexes[:-1, :-1]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc_graph_capacity.extend(upleft_V.reshape(-1).tolist())
    temp1 = img_indexes[1:, :]
    temp2 = img_indexes[:-1, :]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc_graph_capacity.extend(up_V.reshape(-1).tolist())
    temp1 = img_indexes[1:, :-1]
    temp2 = img_indexes[:-1, 1:]
    mask1 = temp1.reshape(-1)
    mask2 = temp2.reshape(-1)
    edges.extend(list(zip(mask1, mask2)))
    gc_graph_capacity.extend(upright_V.reshape(-1).tolist())
    gc_graph = ig.Graph(cols * rows + 2)
    gc_graph.add_edges(edges)
    return gc_graph,gc_source,gc_sink,gc_graph_capacity

def estimate_segmentation(mask,gc_graph,gc_source,gc_sink,gc_graph_capacity,rows,cols):
    mincut = gc_graph.st_mincut(gc_source,gc_sink, gc_graph_capacity)
    print('foreground pixels: %d, background pixels: %d' % (len(mincut.partition[0]), len(mincut.partition[1])))
    pr_indexes = np.where(np.logical_or(mask == DRAW_PR_BG['val'], mask == DRAW_PR_FG['val']))
    img_indexes = np.arange(rows * cols,dtype=np.uint32).reshape(rows, cols)
    mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], mincut.partition[0]),DRAW_PR_FG['val'], DRAW_PR_BG['val'])
    bgd_indexes = np.where(np.logical_or(mask == DRAW_BG['val'],mask == DRAW_PR_BG['val']))
    fgd_indexes = np.where(np.logical_or(mask == DRAW_FG['val'],mask == DRAW_PR_FG['val']))
    print('probble background count: %d, probable foreground count: %d' % (bgd_indexes[0].size,fgd_indexes[0].size))
    return pr_indexes,img_indexes,mask,bgd_indexes,fgd_indexes

def GrabCut(img, mask, rect):
    img = np.asarray(img, dtype=np.float64)
    rows,cols, _ = img.shape
    if rect is not None:
        mask[rect[1]:rect[1] + rect[3],rect[0]:rect[0] + rect[2]] = DRAW_PR_FG['val']

    bgd_indexes = np.where(np.logical_or(mask == DRAW_BG['val'], mask == DRAW_PR_BG['val']))
    fgd_indexes = np.where(np.logical_or(mask == DRAW_FG['val'], mask == DRAW_PR_FG['val']))
    print('probble background count: %d, probable foreground count: %d' % (bgd_indexes[0].size,fgd_indexes[0].size))

    gmm_components = 4
    gamma = 30
    beta = 0
    left_V = np.empty((rows,cols - 1))
    upleft_V = np.empty((rows - 1,cols - 1))
    up_V = np.empty((rows - 1,cols))
    upright_V = np.empty((rows - 1,cols - 1))
    bgd_gmm = None
    fgd_gmm = None
    comp_idxs = np.empty((rows,cols), dtype=np.uint32)
    gc_graph = None
    gc_graph_capacity = None
    gc_source = cols*rows
    gc_sink = gc_source + 1

    _left_diff = img[:, 1:] - img[:, :-1]
    _upleft_diff = img[1:, 1:] - img[:-1, :-1]
    _up_diff = img[1:, :] - img[:-1, :]
    _upright_diff = img[1:, :-1] - img[:-1, 1:]
    sq_left_diff = np.square(_left_diff)
    sq_upleft_diff = np.square(_upleft_diff)
    sq_upright_diff = np.square(_upright_diff)
    sq_up_diff = np.square(_up_diff)
    beta = np.sum(sq_left_diff) + np.sum(sq_upleft_diff) + np.sum(sq_up_diff) + np.sum(sq_upright_diff)
    beta = 1 / (2*beta / (4*cols*rows - 3*cols - 3*rows+ 2))
    print('Beta:',beta)
    left_V = gamma * np.exp(-beta * np.sum(np.square(_left_diff), axis=2))
    upleft_V = gamma / np.sqrt(2) * np.exp(-beta * np.sum(np.square(_upleft_diff), axis=2))
    up_V = gamma * np.exp(-beta * np.sum(np.square(_up_diff), axis=2))
    upright_V = gamma / np.sqrt(2) * np.exp(-beta * np.sum(np.square(_upright_diff), axis=2))

    bgd_gmm = GaussianMixture(img[bgd_indexes])
    fgd_gmm = GaussianMixture(img[fgd_indexes])

    gc_graph,gc_source,gc_sink,gc_graph_capacity = construct_gc_graph(img,mask,gc_source,gc_sink,fgd_gmm,bgd_gmm,gamma,rows,cols,left_V,upleft_V,up_V,upright_V)
    pr_indexes,img_indexes,mask,bgd_indexes,fgd_indexes = estimate_segmentation(mask,gc_graph,gc_source,gc_sink,gc_graph_capacity,rows,cols)
    return mask

BLUE = [255, 0, 0]

DRAW_BG = {'val': 0}
DRAW_FG = { 'val': 1}
DRAW_PR_FG = {'val': 3}
DRAW_PR_BG = {'val': 2}

rect = (0, 0, 1, 1)
drawing = False
rectangle = False
rect_over = False
rect_or_mask = 100
value = DRAW_FG
flag = True

def onmouse(event, x, y, flags, param):
    global img
    global img2
    global drawing
    global value
    global mask
    global rectangle
    global rect
    global rect_or_mask
    global ix, iy
    global rect_over
    global flag
    if event == cv.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE and rectangle == True:
        img = img2.copy()
        cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
        rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
        rect_or_mask = 0

    elif event == cv.EVENT_RBUTTONUP and flag == True:
        rectangle = False
        rect_over = True
        cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
        rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")

    if event == cv.EVENT_LBUTTONDOWN and rect_over == False and flag == True:
        print("first draw rectangle \n")


if __name__ == '__main__':

    print('Interactive Image Segmentation using GrabCut algorithm')
    print('-------------------------------------------------------')
    print('The Input window contains the original image')
    print('The Output window contains the output of grabcut algorithm ')
    print('In the begining draw a rectange around the foreground object using right mouse button')
    print('Then press "g" button to segement the foreground object')
    print('For Finer Touch ups:')
    print("Key 'b' - To select areas that you are sure belongs to background")
    print("Key 'f' - To select areas that you are sure belongs to foreground")
    print("Key 'a' - To select areas that probably belongs to background")
    print("Key 'd' - To select areas that probably belongs to foreground")
    print("Key 'g' - To update the segmentation")
    print("Key 'r' - To reset the setup")
    print("Key 's' - To save the results")
    print('NOTE: If the argument contains a textfile with bounding box')
    print('so just press g without drawing a bounding box')
    if len(sys.argv) == 3:
        filename = sys.argv[1]
        file = open(sys.argv[2],'r')
        line = file.readlines()
        prerect = line[0].split(' ')
        prerect = list(map(int, prerect))
        print(prerect)
    else:
        filename = sys.argv[1]

    img = cv.imread(filename)
    img2 = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    output = np.zeros(img.shape, np.uint8)

    cv.namedWindow('output')
    cv.namedWindow('input')
    cv.setMouseCallback('input', onmouse)
    cv.moveWindow('input', img.shape[1]+10, 90)

    def reset():
        print("resetting \n")
        rect = (0, 0, 1, 1)
        drawing = False
        rectangle = False
        rect_or_mask = 100
        rect_over = False
        value = DRAW_FG

    while(1):
        cv.imshow('output', output)
        cv.imshow('input', img)
        k = cv.waitKey(1)
        if k == 27:
            break
        elif k == ord('b') and flag == True:
            print(" mark background regions with left mouse button \n")
            value = DRAW_BG
        elif k == ord('f')and flag == True:
            print(" mark foreground regions with left mouse button \n")
            value = DRAW_FG
        elif k == ord('a') and flag == True:
            value = DRAW_PR_BG
        elif k == ord('d') and flag == True:
            value = DRAW_PR_FG
        elif k == ord('s') and flag == True:
            bar = np.zeros((img.shape[0], 5, 3), np.uint8)
            res = np.hstack((img2, bar,img,bar, output))
            cv.imwrite('output.png', res)
            print(" Result saved as image \n")
        elif k == ord('r'):
            reset()
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            output = np.zeros(img.shape, np.uint8)
        elif k == ord('g'):
            if len(sys.argv) == 3:
                rect = prerect
                mask = GrabCut(img2,mask,rect)
            else:
                print(rect)
                mask = GrabCut(img2, mask, rect)

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        output = cv.bitwise_and(img2, img2, mask=mask2)

    cv.destroyAllWindows()
