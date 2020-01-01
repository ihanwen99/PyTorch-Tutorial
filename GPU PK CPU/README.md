# GPU PK CPU

这是一个GPU和CPU进行比较的代码。

在`Cifer-10`数据集上测试`VGG19`相对复杂的网络。

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

```

```

