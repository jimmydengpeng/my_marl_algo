import time
import gym 
import highway_env
from colorlog import logger
from tqdm import tqdm

def main():
    env = gym.make("highway-v0")
    obs = env.reset()
    done = False
    step = 0
    t0 = time.time()
    MAX_STEP = 100
    # while not done:
    for i in tqdm(range(MAX_STEP)):
        action = env.action_space.sample()
        # action = 0.0
        # print(action)
        obs, reward, done, info = env.step(action)
        # env.render()
        step += 1
        # logger.debug("step:", step, inline=True)

        if done:
            done = False
            env.reset()
            # logger.info("RESET")
    
    t1 = time.time()
    logger.success("average step:", (MAX_STEP/(t1-t0)), inline=True)


if __name__ == "__main__":
    main()