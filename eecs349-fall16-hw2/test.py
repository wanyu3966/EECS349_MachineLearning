# location, last place to login, like someone count, age, gender,  
# metric: distance feature

#导入pyplot包，并简写为plt
import matplotlib.pyplot as plt


#将绘画框进行对象化
fig=plt.figure()

#将p1定义为绘画框的子图，211表示将绘画框划分为2行1列，最后的1表示第一幅图
p1=fig.add_subplot(211)
x=[1,2,3,4,5,6,7,8]
y=[2,1,3,5,2,6,12,7]
p1.plot(x,y)

#将p2定义为绘画框的子图，211表示将绘画框划分为2行1列，最后的2表示第二幅图
p2=fig.add_subplot(212)
a=[1,2]
b=[2,4]
p2.scatter(a,b)

plt.show()

作者：挖数
链接：https://zhuanlan.zhihu.com/p/21443208
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。


#导入pyplot包，并简写为plt
import matplotlib.pyplot as plt

#导入3D包
from mpl_toolkits.mplot3d import Axes3D

#将绘画框进行对象化
fig = plt.figure()

#将绘画框划分为1个子图，并指定为3D图
ax = fig.add_subplot(111, projection='3d')

#定义X,Y,Z三个坐标轴的数据集
X = [1, 1, 2, 2]
Y = [3, 4, 4, 3]
Z = [1, 100, 1, 1]

#用函数填满4个点组成的三角形空间
ax.plot_trisurf(X, Y, Z)
plt.show()

作者：挖数
链接：https://zhuanlan.zhihu.com/p/21443208
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#def stringToWords(string1):
 #   parts = [''.join(c for c in s if c.isalpha()) for s in string1.split()]
  #  return parts

#print stringToWords('aaa 123bbb cc123`ee fff')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt(
                        'expResult.txt',      # file name
                        delimiter='\t',         # column delimiter
                        dtype='float',            # data type
                        #usecols = (0,1,2,3),      # use firs 4 columns only
                        names=['deletion_cost','insertion_cost','substitution_cost', 'error_rate']     # column names
                        )

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_title('problem 3')
ax.set_xlabel('deletion_cost')
ax.set_ylabel('insertion_cost')
ax.set_zlabel('error_rate')

ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
ax.set_zlim(0, 1)

ax.view_init(elev=12, azim=40)              # elevation and angle
ax.dist=12                                  # distance

ax.scatter(
           data['deletion_cost'], data['insertion_cost'], data['error_rate'],  # data
           color='purple',                            # marker colour
           marker='o',                                # marker shape
           s=30                                       # marker size
           )
print data
plt.show()                                            # render the plot
