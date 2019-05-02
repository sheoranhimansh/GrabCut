# GrabCut

My implementation of *[Grabcut: Interactive foreground extraction using iterated graph cuts, ACM SIGGRAPH 2004](https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf)*, without border matting.

## Dependencies

* Python 3.6
* OpenCV
* NumPy
* scikit-learn
* python-igraph

## File desctiptions

* `grabcut.py` - Core implementation of the algorithm.
* `messi5.jpg` - A copy of [OpenCV's sample image](https://github.com/opencv/opencv/blob/master/samples/data/messi5.jpg).

## Usage

```
python grabcut.py <filename>
```
