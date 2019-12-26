# 12306 验证码训练

最近在学习DNN，有些坑，搞个仓库记录一下原因

### 1.训练集

首先数据量一定要大，不然就会出现过拟合严重的情况，有的时候甚至train的loss不断下降，但是test的丝毫不动。在这次的case里面因为dataset只有大约1万多张图片，远远不够，倒是训练过程中，train的loss一直比较稳定。需要进行相关的图像增强

### 2.keras的Sequential

这个类有好多的构造函数，导致构造的过程中会出现Layer叠加的问题，比如
```python
model = Sequential([
Conv2D(input_shape=(ROW, COL, 3), filters=32, kernel_size=(3, 3), padding='same', activation='relu',
           data_format='channels_last'),
           ...
```

```python
model = Sequential([
Conv2D(input_shape=(ROW, COL, 3), filters=32, kernel_size=(3, 3), padding='same',data_format='channels_last'),
Activation('relu'),
	...
```

以上两个构造Layer的方式，区别在于如果在Sequential里面初始化层的话，需要在每个layer里面指定对应的Activation Function（第一个code片段），否则会出现问题（具体原因不知，实验结果如此）

### 3.训练速度

由于没有什么条件原先的时候用Mac Pro硬跑的，结果死的很惨，一个epcho需要大概15分钟（6百多万的参数，而且电脑风扇不停的转😂），后来Google了一下，发现了大法提供了两个平台可以用GPU甚至TPU来跑对应的代码。这两个分别是colab和kaggle。最初一直在体验colab，感觉需要挂在Google的driver来获取数据(当然也有其他的方式如github)。总体感觉发现了新大陆，不过尝试了一下发现速度还没有Mac Pro快，后来看了一下发现主要是因为异步加载数据的方式实在是太慢了，后来索性在本地把dataset做好，然后上传上去，一次性load到内存，这个速度果然是杠杆的。大概一个epcho只需要27s的样子。TPU没有做尝试，需要进行模型转换，据说更加快，回头研究一下。
后来试了一下kaggle，然后就有了一定的比较。dataset的方面，kaggle比较好，可以直接网上传，读取也是非常方便的。但是在资源方面有所不足，没有TPU，此外GPU使用也是有限制的，一周只能用30个小时的样子。内存方面的话最高是13G（colab可以在一次崩溃之后切到高内存模式，最大25G的样子，且没有使用时间的限制）。

### 4.拟合相关

一开始训练发现train的loss下降，但是测试的loss一直不下降，甚至还有上升。很不理解，看上去像过拟合。尝试了BN和Regular还有drop都没有效果。后来增强数据集之后有明显改善。样本的大小对于训练的效果有非常大的作用，后续还是要在数据集的方面去多考虑，毕竟这个是最有效的手段。

### 5.内存
python的内存管理不是及时释放的，在生成增强数据保存的时候出现了内存过大被kill的情况。还是需要及时的释放内存
使用
```python
import gc
del var
gc.collect()
```
### 6.keras的history 
```python
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Model Accuracy")
key = 'accuracy'
val_key = 'val_accuracy'
if 'acc' in history.history.keys():
    key = 'acc'
    val_key = 'val_acc'

plt.plot(history.history[key])
plt.plot(history.history[val_key])
```
在mac上的keyword是‘accuracy’，但是在colab上是‘acc’



### 后续
训练完成之后发生了一件非常沙雕的事情，acc达到了99.7%，但是验证图片全部是同一个idx，一度怀疑是训练集数据有问题，查了好久之后发现原来输入的验证图片没有进行归一化😂


### 附上数据集
增强图片数据集:https://www.kaggle.com/leehomwang/12306-data

图片:https://www.kaggle.com/libowei/12306-captcha-image
