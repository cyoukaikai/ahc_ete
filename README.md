# AHC_ETE

This repository includes the code used for the paper "From Certain to Uncertain: Toward Optimal Solution for Offline Multiple Object Tracking" (ICPR 2020).
The work was done when I am working at Smart Mobility Reseach Center (SMRC) of Tokyo University of Agriculture and Technology.
So I named the library `smrc`.

### Installation

```
conda create -n ahc-ete python=3.7
conda activate ahc-ete

pip install -r requirements.txt

```

We modified the [`scikit-learn 0.23.2`](https://github.com/scikit-learn/scikit-learn) library for the `linkages`
used in our method.
We only modified the file [`scikit-learn/sklearn/cluster/_hierarchical_fast.pyx`](scikit-learn/sklearn/cluster/_hierarchical_fast.pyx).

In that file, we changed `from libc.math cimport fmax` to `from libc.math cimport fmax, fmin` and
we added `max_merge_tracking`, `average_merge_tracking`, 
`single_merge_tracking`, `single_merge` functions.


To install the modified `scikit-learn` library, 

```
cd scikit-learn
pip install -e ./
cd ../
```


### Downloading MOTChallenge dataset

The `MOTChallenge` datasets can be downloaded from its [official site](https://motchallenge.net/). We assume you
have downloaded the [`MOT16`](https://motchallenge.net/data/MOT16/) and 
[`MOT15`](https://motchallenge.net/data/2D_MOT_2015/) sequences in `./MOTChallenge/MOT16` and `./MOTChallenge/MOT15`, respectively. 

### Generating features for public detections

We use the `deep_sort` library to conduct this task.
The `deep_sort` library included in this repo was downloaded from its official implementation [here](https://github.com/nwojke/deep_sort) 
in April, 2020.

We will obtain the features for `MOT16` and `MOT15` by running 
```
cd deep_sort
python tools/generate_detections.py \
    --model=resources/networks/mars-small128.pb \
    --mot_dir=../MOTChallenge/MOT16/train \
    --output_dir=./resources/detections/MOT16_train

python tools/generate_detections.py \
    --model=resources/networks/mars-small128.pb \
    --mot_dir=../MOTChallenge/MOT15/train \
    --output_dir=./resources/detections/MOT15_train
```
To get the tracking results, run 
```
python evaluate_motchallenge.py --mot_dir=../MOTChallenge/MOT16/train --detection_dir=./resources/detections/MOT16_train --output_dir ./MOT16_train_results --min_confidence=0.3 --nn_budget=100
```

In our experiments, we restrict both our method and [Deep Sort](https://arxiv.org/abs/1703.07402) to merge only existing detections 
(without handling missed detections) for comparison. The command used for Deep Sort is,

```
python evaluate_motchallenge_no_prediction.py --mot_dir=../MOTChallenge/MOT16/train --detection_dir=./resources/detections/MOT16_train --output_dir ./MOT16_train_results --min_confidence=0.3 --nn_budget=100
```

To conduct tracking using  our method (`AHC_ETE`), run  

```
cd ../
ln -s $PWD/deep_sort/resources ./MOTChallenge/resources
cd smrc/object_tracking/benchmark/
python mot_eval.py
```

The following section is where we set the used tracking experts and the test datasets in `mot_eval.py`.
```
if __name__ == "__main__":
    mot = MOTEval()

    from smrc.object_tracking.benchmark.expertconfig_ahc_ete import ExpertTeam
    mot.evaluate_AHC_ETE('MOT16/train', expert_team_config=ExpertTeam)
    mot.evaluate_AHC_ETE('MOT15/train', expert_team_config=ExpertTeam)

```
The results are saved in `smrc/object_tracking/benchmark/MOT16` and `smrc/object_tracking/benchmark/MOT15`.

### Evaluating the tracking results

The evaluation code was downloaded from https://github.com/cheind/py-motmetrics in October, 2020.
First, we need to install the dependencies.
```
cd py-motmetrics-develop
pip install -r requirements.txt
```

Then, we can evaluate the results by
```
python -m motmetrics.apps.eval_motchallenge ../MOTChallenge/MOT16/train ../smrc/object_tracking/benchmark/MOT16
```
We obtained the following results,
```
05:16:01 INFO - Found 7 groundtruths and 7 test files.
05:16:01 INFO - Available LAP solvers ['scipy']
05:16:01 INFO - Default LAP solver 'scipy'
05:16:01 INFO - Loading files.
05:16:02 INFO - Comparing MOT16-10...
05:16:03 INFO - Comparing MOT16-13...
05:16:03 INFO - Comparing MOT16-05...
05:16:04 INFO - Comparing MOT16-02...
05:16:05 INFO - Comparing MOT16-11...
05:16:05 INFO - Comparing MOT16-04...
05:16:06 INFO - Comparing MOT16-09...
05:16:07 INFO - Running metrics
05:16:08 INFO - partials: 1.135 seconds.
05:16:08 INFO - mergeOverall: 1.142 seconds.
          IDF1   IDP   IDR  Rcll  Prcn  GT MT  PT  ML   FP    FN IDs    FM  MOTA  MOTP IDt IDa IDm
MOT16-10 39.7% 59.8% 29.7% 40.9% 82.3%  54  7  22  25 1085  7283  99   436 31.3% 0.255  22  52  10
MOT16-13 30.2% 70.5% 19.2% 23.6% 86.5% 107 10  33  64  422  8726  47   240 19.5% 0.271  21  34  15
MOT16-05 46.9% 78.8% 33.4% 39.0% 92.0% 125  9  57  59  229  4140  57   153 34.8% 0.242  22  38  17
MOT16-02 33.3% 77.0% 21.2% 23.4% 84.9%  54  6  15  33  741 13658  60   283 18.9% 0.249  26  22   7
MOT16-11 61.3% 80.5% 49.5% 55.9% 91.0%  69 12  25  32  507  4047  38    94 49.9% 0.215   7  26   4
MOT16-04 46.7% 67.0% 36.2% 45.1% 82.1%  83  9  42  32 4692 26086 191   983 34.9% 0.221  91  72   9
MOT16-09 49.1% 55.6% 44.0% 58.1% 73.5%  25  4  17   4 1102  2203 102   164 35.2% 0.265  36  46   2
OVERALL  44.1% 68.5% 32.6% 40.1% 83.4% 517 57 211 249 8778 66143 594  2353 31.6% 0.234 225 290  64
05:16:08 INFO - Completed
```

To evaluate the results of Deep Sort,
```
python -m motmetrics.apps.eval_motchallenge ../MOTChallenge/MOT16/train  ../deep_sort/MOT16_train_results
```

### Reference 
The first paper is for Deep Sort and the second paper is for our method.
```
@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756}
}
@inproceedings{zhao2020from,
  title={From Certain to Uncertain: Toward Optimal Solution for Offline Multiple Object Tracking},
  author={Zhao, Kaikai and Imaseki, Takashi and Mouri, Hiroshi and Suzuki, Einoshin and Matsukawa, Tetsu},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  year={2020}
}

```