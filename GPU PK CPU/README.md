# GPU PK CPU

这是一个GPU和CPU进行比较的代码。

在`Cifer-10`数据集上测试`VGG19`相对复杂的网络。

BatchSize=4【突然发现这个太少，也有可能影响我的计算速率】

GPU使用`GTX1050Ti`，CPU是`Intel 7700HQ`。

---

**经过一次迭代，GPU的结果是：**

```
"""
GTX1050Ti

cuda:0
Finished Training
Time for 2000 iters: 252.42594933509827
Total Time is 1571.9463894367218
GroundTruth:    cat  ship  ship plane
Predicted:    cat  ship  ship  ship

Accuracy of the network on the 10000 test images: 39 %
Accuracy of plane :  0 %
Accuracy of   car :  0 %
Accuracy of  bird : 25 %
Accuracy of   cat : 24 %
Accuracy of  deer :  0 %
Accuracy of   dog :  0 %
Accuracy of  frog : 26 %
Accuracy of horse : 25 %
Accuracy of  ship :  0 %
Accuracy of truck :  0 %

Process finished with exit code 0

"""
```

**经过一次迭代，CPU的结果是：**

这个其实就不用再继续实验对比了，CPU的速度的确相对GPU的速度差了很多。

```
[1, 1, 2.30128812789917]
[1, 2, 2.3028969764709473]
[1, 3, 2.291522979736328]
[1, 4, 2.299872398376465]
[1, 5, 2.2974934577941895]
[1, 6, 2.3089444637298584]
[1, 7, 2.3088276386260986]
[1, 8, 2.3043880462646484]
[1, 9, 2.2943291664123535]
[1, 10, 2.3111562728881836]

[1, 3168, 2.3857061862945557]
[1, 3169, 2.8947196006774902]
[1, 3170, 1.6989765167236328]
Time for 2000 iters: 2346.870943069458
```

---

**这个是使用服务器训练的Loss显示：**

第一次数据是`batch_size=4`，训练`epoch=10`

```
[1,  2000] loss: 2.237
[1,  4000] loss: 2.072
[1,  6000] loss: 1.912
[1,  8000] loss: 1.832
[1, 10000] loss: 1.736
[1, 12000] loss: 1.704
[2,  2000] loss: 1.612
[2,  4000] loss: 1.542
[2,  6000] loss: 1.493
[2,  8000] loss: 1.419
[2, 10000] loss: 1.363
[2, 12000] loss: 1.275
[3,  2000] loss: 1.199
[3,  4000] loss: 1.127
[3,  6000] loss: 1.107
[3,  8000] loss: 1.085
[3, 10000] loss: 1.053
[3, 12000] loss: 1.026
[4,  2000] loss: 0.928
[4,  4000] loss: 0.901
[4,  6000] loss: 0.881
[4,  8000] loss: 0.890
[4, 10000] loss: 0.868
[4, 12000] loss: 0.855
[5,  2000] loss: 0.752
[5,  4000] loss: 0.761
[5,  6000] loss: 0.738
[5,  8000] loss: 0.722
[5, 10000] loss: 0.732
[5, 12000] loss: 0.727
[6,  2000] loss: 0.599
[6,  4000] loss: 0.647
[6,  6000] loss: 0.653
[6,  8000] loss: 0.643
[6, 10000] loss: 0.639
[6, 12000] loss: 0.614
[7,  2000] loss: 0.532
[7,  4000] loss: 0.520
[7,  6000] loss: 0.542
[7,  8000] loss: 0.552
[7, 10000] loss: 0.549
[7, 12000] loss: 0.542
[8,  2000] loss: 0.440
[8,  4000] loss: 0.457
[8,  6000] loss: 0.486
[8,  8000] loss: 0.449
[8, 10000] loss: 0.481
[8, 12000] loss: 0.489
[9,  2000] loss: 0.351
[9,  4000] loss: 0.422
[9,  6000] loss: 0.388
[9,  8000] loss: 0.418
[9, 10000] loss: 0.409
[9, 12000] loss: 0.419
[10,  2000] loss: 0.308
[10,  4000] loss: 0.368
[10,  6000] loss: 0.360
[10,  8000] loss: 0.353
[10, 10000] loss: 0.370
[10, 12000] loss: 0.361
Finished Training

Accuracy of the network on the 10000 test images: 78 %

Accuracy of plane : 88 %
Accuracy of   car : 90 %
Accuracy of  bird : 64 %
Accuracy of   cat : 63 %
Accuracy of  deer : 83 %
Accuracy of   dog : 59 %
Accuracy of  frog : 82 %
Accuracy of horse : 79 %
Accuracy of  ship : 86 %
Accuracy of truck : 88 %
```

改大batch_size了之后，很快就训练完了新的50次迭代。