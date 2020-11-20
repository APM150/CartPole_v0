from Learner import Learner
import torch

if __name__ == '__main__':
    learner = Learner(0.001, "Model/goodPolicy.pth")
    # train
    # for epoch in range(1000):
    #     learner.trainPolicy(1000, 0.2)

    # simulate
    result = learner.simulate(1, 1, True)
    for i in result:
        print(i[2])
