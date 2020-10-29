# AHC_ETE

This repository includes the code used for the paper "From Certain to Uncertain: Toward Optimal Solution for Offline Multiple Object Tracking" (ICPR 2020).
The work was done when I am working at Smart Mobility Reseach Center (SMRC) of Tokyo University of Agriculture and Technology.


### Installation

```
conda create -n ahc-ete python=3.7
conda activate ahc-ete

pip install -r requirements.txt

```

We modified the [`scikit-learn 0.23.2`](https://github.com/scikit-learn/scikit-learn) library for the linkage
used in our method.
We only modified the file [`scikit-learn/sklearn/cluster/_hierarchical_fast.pyx`](scikit-learn/sklearn/cluster/_hierarchical_fast.pyx).

We changed `from libc.math cimport fmax` to `from libc.math cimport fmax, fmin` and
we added `max_merge_tracking`, `average_merge_tracking`, 
`single_merge_tracking`, `single_merge` functions.
To install the modified `scikit-learn` library, 

```
cd scikit-learn
pip install -e ./
cd ../
```


### Conduct tracking


### Recurrent Neural Networks

* [Deep learning book](http://www.deeplearningbook.org/) 
[Chapter [10 Sequence Modeling: Recurrent and Recursive Nets](http://www.deeplearningbook.org/contents/rnn.html)]
* [All of Recurrent Neural Networks (Notes of the deep learning book)](https://medium.com/@jianqiangma/all-about-recurrent-neural-networks-9e5ae2936f6e)

