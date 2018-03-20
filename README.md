# Lifelong Learning with Dynamically Expandable Networks
+ Jaehong Yoon(UNIST), Eunho Yang(KAIST), Jeongtae Lee(UNIST), and Sung Ju Hwang(UNIST)

We propose a novel deep network architecture for lifelong learning which we refer to as Dynamically Expandable Network (DEN), that can dynamically decide its network capacity as it trains on a sequence of tasks, to learn a compact overlapping knowledge sharing structure among tasks. DEN is efficiently trained in an online manner by performing selective retraining, dynamically expands network capacity upon arrival of each task with only the necessary number of units, and effectively prevents semantic drift by splitting/duplicating units and timestamping them. We validate DEN on multiple public datasets in lifelong learning scenarios on multiple public datasets, on which it not only significantly outperforms existing lifelong learning methods for deep networks, but also achieves the same level of performance as the batch model with substantially fewer number of parameters. 

## Reference

If you use this code as part of any published research, please refer the following paper. [DEN](https://arxiv.org/abs/1708.01547/)

```
@inproceedings{
    yoon2018lifelong,
    title={Lifelong Learning with Dynamically Expandable Networks},
    author={Jaehong Yoon and Eunho Yang and Jeongtae Lee and Sung Ju Hwang},
    booktitle={International Conference on Learning Representations},
    year={2018},
    url={https://openreview.net/forum?id=Sk7KsfW0-},
}
```

## Running Code

We implemented the model as described in the paper based on Tensorflow library, [Tensorflow](https://www.tensorflow.org/).

### Get our code
```
git clone --recursive https://github.com/jaehong-yoon93/DEN.git DEN
```

### Run examples

In this code, you can run our model on MNIST dataset with permutation. Then, you don't need to download dataset on your own, just you get the dataset when you run our code.

For convinence, we added the logs that are printed out validation & test accuracy, and several process.
If you execute DEN_run.py, you can reproduce our model.  

```
python DEN_run.py
```

## Authors

[Jaehong Yoon](https://jaehongyoon.wordpress.com/)<sup>1</sup>, [Eunho Yang](https://sites.google.com/site/yangeh/)<sup>2</sup><sup>3</sup>, [Jeongtae Lee](https://github.com/jeong-tae)<sup>1</sup>, and [Sung Ju Hwang](http://www.sungjuhwang.com/)<sup>1</sup><sup>3</sup>

<sup>1</sup>[MLVR Lab](http://ml.unist.ac.kr/) @ School of Electrical and Computer Engineering, UNIST, Ulsan, South Korea

<sup>2</sup>[KAIST](http://www.kaist.edu/) @ School of Computing, KAIST, Daejeon, South Korea

<sup>3</sup>[AItrics](https://www.aitrics.com/) @ Seoul, South Korea
