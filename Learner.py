from Policy import Policy
import gym
import random
import torch
import torch.nn as nn


class Learner:
    def __init__(self, learning_rate=0.01, FILE="Model/goodPolicy.pth"):
        self.FILE = FILE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = Policy().to(self.device)
        self.policy.load_state_dict(torch.load(self.FILE))
        self.policy.eval()
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)


    def simulate(self, episode: int, policyPercent: float, show=False):
        """
        Simulate the cartpole process
        :param episode: number of episode want to simulate, how many percentage of policy want to use
        :return: list of ([trajectory of actions], [trajectory of observation], totalReward)
        """
        env = gym.make('CartPole-v0')
        result = []
        for i_episode in range(episode):
            actions = []
            observations = []
            totalReward = 500  # if not failed
            observation = env.reset()
            for t in range(500):
                if show: env.render()
                observationTensor = torch.from_numpy(observation)  # convert from numpy to tensor
                observationTensor = torch.tensor(observationTensor, dtype=torch.float32)
                observationTensor = observationTensor.to(self.device)
                observations.append(observation.tolist())
                if random.random() <= policyPercent:  # policy mix with random choice
                    with torch.no_grad():
                        action = torch.max(self.policy(observationTensor), 0)[1].item()  # 0 or 1
                else:
                    action = random.randint(0, 1)
                actions.append(action)
                observation, reward, done, info = env.step(action)
                if done:
                    totalReward = t + 1
                    # print(f"Episode finished after {t + 1} timesteps")
                    break
            result.append((actions, observations, totalReward))
        env.close()
        return result


    def trainPolicy(self, episodes, policyPercent=0.8):
        """ Train the policy """
        # First play serval times to determine the average reward.
        trajectoriesForAvgRwd = self.simulate(20, 1)
        averageReward = sum([i[2] for i in trajectoriesForAvgRwd]) / len(trajectoriesForAvgRwd)
        print(averageReward)

        trajectoriesForTrain = self.simulate(episodes, policyPercent)
        for trainTrajectory in trajectoriesForTrain:
            if trainTrajectory[2] > averageReward:
                # forward
                predictAction = self.policy(torch.tensor(trainTrajectory[1]).to(self.device))
                loss = self.criterion(predictAction, torch.tensor(trainTrajectory[0]).to(self.device))

                # backwards
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        torch.save(self.policy.state_dict(), self.FILE)
