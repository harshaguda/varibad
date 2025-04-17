from clothsuite.envs.dressing_v4 import DressingEnv
import random 
import numpy as np

class DressingElbowEnv(DressingEnv):
    """Dressing environment with elbow target direction."""

    def __init__(self, max_episode_steps=1024):
        self.task_dim = 1
        self._max_episode_steps = max_episode_steps
        super(DressingElbowEnv, self).__init__()
        self.set_task(self.sample_tasks(1)[0])
    
    def step(self, action):
        return super().step(action)
    
    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        return [random.choice([0, 90]) for _ in range(n_tasks, )]
    
    def set_task(self, task):
        self.goal_elbow = task
        print("Elbow task: ", task)
        # self.make_human(task)

    def get_task(self):
        return np.array([self.goal_elbow])
    
    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)
    
    def seed(self, seed=None):
        return