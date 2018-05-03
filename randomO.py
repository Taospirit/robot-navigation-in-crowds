import random
from GameClass import width,height

obs_radius = 30
ran=[]
for i in range(10):
    ran.append([random.randint(obs_radius, width - obs_radius),random.randint(obs_radius, height - obs_radius)])
print(ran)
# [[390, 774], [917, 349], [660, 580], [730, 344], [712, 204], [431, 516], [1048, 199], [1155, 689], [660, 134], [826, 589]]