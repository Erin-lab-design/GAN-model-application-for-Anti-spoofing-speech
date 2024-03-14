这个代码就是抄 https://github.com/pquochuy/idsegan 中的SEGAN部分改的，去掉了做speech enhancement（SE，降噪）部分，改了discriminator部分的损失函数（改成EER）来做spoofing speech detection。
但是我突然发现了 https://zenodo.org/records/6635521 这个数据集，对应的 https://github.com/ADDchallenge/CFAD 这个代码库。
用GPT的话来说就是：这篇论文的核心内容是设计和介绍了一个中文虚假音频检测数据集（FAD），旨在研究更通用的检测方法，以应对当前增长的虚假音频检测挑战。
这篇论文的主要目标是填补中文虚假音频检测数据集的空白，为研究更通用化的检测方法提供基础。当前存在的数据集大多聚焦于特定语言或条件，缺乏适用于中文且考虑添加噪声条件下的公开数据集。

这不巧了吗？https://github.com/pquochuy/idsegan 本来就是做降噪的呀！
所以我就想完全可以保留SE部分功能，加上discriminator鉴别真假语音的功能，这不就创新了解决重大问题了嘛。

还有更简单的方法就是直接把 https://zenodo.org/records/6635521 数据集中的合成加噪语音给降噪了，再喂给 https://github.com/ADDchallenge/CFAD 做spoofing detection，效果肯定会更好啊。

但这样就显得怎么说，太简单了，太敷衍了，当我们不是不可以把SEGAN模型和用于spoofing detection的模型连在一起用，再给这个brand new model起个很厉害的名字，听起来就像那么回事了呢。

最后，如果你比对 https://github.com/pquochuy/idsegan 中SEGAN部分代码和我这的，就会发现我没抄完，还有几个没上传的就是还没改。已经上传了的，也有一些细节，比如说noise speech --> fake speech啊什么的，名字没改过来。

我会上传一些参考文献到一个特定的文件夹里。
