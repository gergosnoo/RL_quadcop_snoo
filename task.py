import numpy as np
from physics_sim import PhysicsSim
from math import pow

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=3., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 4

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        reward = 0
        """Uses current pose of sim to return reward."""
        # reward = 1. - .35 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        # reward = 1-.1*abs(self.sim.pose[2] - self.target_pos[2])

        # if 10. - (abs(self.sim.pose[:3] - self.target_pos)).sum() <= 5:
        #     reward = 0
        # else:
        #     reward = 10. - (abs(self.sim.pose[:3] - self.target_pos)).sum()

        if abs(self.sim.pose[:3] - self.target_pos).sum() < 1.5:
            reward += 5

        # if abs(self.sim.pose[2] - self.target_pos[2]) < 5:
        #     reward += (- .2 * pow((self.sim.pose[2] - self.target_pos[2]), 2) + 10)
        # if break1 <= abs(self.sim.pose[2] - self.target_pos[2]) < 10:
        #     reward = .04 * pow(((abs(self.sim.pose[:3] - self.target_pos)).sum())-10, 2) - 1

        if self.target_pos[2]+3 >= self.sim.pose[2]:
            reward += 10.0

        if (abs(self.sim.pose[:3] - self.target_pos)).sum() <= 0.75:
            reward += 10.0

        done = False
        rotor = np.array([0., 0., 0., 0.])

        if self.sim.pose[2] == self.target_pos[2]:  # agent has crossed the target low
            # raise TypeError
            reward += 50.0  # bonus reward
            if np.all(self.sim.prop_wind_speed == rotor):
                reward += 100.
            done = True

        return reward, done

        # return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            delta_reward, done_land = self.get_reward()
            reward += delta_reward
            pose_all.append(self.sim.pose)
            if done_land:
                done = done_land
                # return np.concatenate([self.sim.pose] * self.action_repeat), reward, done
        next_state = np.concatenate(pose_all)

        # if done and self.sim.time < self.sim.runtime:
        #     reward = -2000

        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state