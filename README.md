# Fetchreach_gym
I have DDPG+HER to train FetchReach-V1. After training for 5 hours, its accuracy is around 90% to 95%.

## Result after training for 5 hours
![new_performance_ddpg_her](https://user-images.githubusercontent.com/28859302/43928989-b457a0b8-9c50-11e8-92cc-3db032b251d5.png)

column = [" Episode_no ",  " reward for the current episode ",  " average reward ",  " Max_reward "]

Average is taken over last 10 observation.

Every Episode contains 200 actions to reach the goal. You would receive a reward of 0 if you reached the goal else 0. So, max and min reward for an episode is 0 and -200.

#### I have only sampled given achieve goals and desired goal, but you can also sample different goal to make algorithm learn faster. 
