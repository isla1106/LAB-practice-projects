# LAB-practice-projects
including cifar10，ResNet，IMDB，LoRA

## Practice1
构建深度为7层的CNN模型在CIFAR10数据集上训练，模型包括四层卷积层和三层全连接层。不同超参数设置下训练效果如下表所示。
| epoch | optimizer | learning rate | batch size | accuracy|
| :----- 	| :--: | :--: | :--: | -------:	 |
| 50 	|  Adam  | 0.001 | 128 | 72% |
| 50 	|  Adam  | 0.01 | 128 | 57% |
| 100 	|  Adam  | 0.0001 | 128 | 69% |
| 100 	|  Adam  | 0.001 | 64 | 71% |
| 100 	|  Adam  | 0.001 | 256 | 71% |
| 100 	|  Adam  | 0.001 | 512 | 70% |
| 100 	|  SGD  | 0.001 | 128 | 68% |


## Practice2
复现 ResNet-18 模型，分别在移除与保留残差连接的模型上使用CIFAR10数据集训练该模型.训练效果如下表所示。

| 残差连接 | epoch | optimizer | learning rate | batch size | accuracy|
| :----- 	| :--: | :--: | :--: | :--: | -------:	 |
| 移除 | 10 	|  Adam  | 0.001 | 128 | 72% |
| 保留 | 10 	|  Adam  | 0.001 | 128 | 73% |

模型在各个类别的准确率如下。
total : 73 %
plane : 82 %
car : 78 %
bird : 69 %
cat : 57 %
deer : 56 %
dog : 54 %
frog : 85 %
horse : 75 %
ship : 82 %
truck : 78 %




