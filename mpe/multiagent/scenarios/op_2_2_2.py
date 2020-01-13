import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents = 2
        world.num_goals = num_goals = 2
        world.num_obstacles = num_obstacles = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_goals+num_obstacles)]
        for i, landmark in enumerate(world.landmarks):
            if i < num_goals:
                landmark.name = 'goal %d' % i
                landmark.collide = True
                landmark.movable = False
            else:
                landmark.name = 'obstacle %d' % (i-num_goals)
                landmark.size = 0.1
                landmark.collide = False
                landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_ang = 2 * np.pi * np.random.uniform(0, 1)
            agent.state.p_vel = np.array([1e-5 * np.cos(agent.state.p_ang), 1e-5 * np.sin(agent.state.p_ang)]) #np.zeros(world.dim_p)
            #agent.state.p_vel = np.zeros(world.dim_p)
            #agent.state.phi = 0
            #agent.state.avel = 0#np.zeros(world.dim_p)
            #agent.state.p_vel = np.random.uniform(0, +1, world.dim_p)
            #agent.state.acc = 0#np.zeros(world.dim_p)
            #agent.state.ag_acc = 0#np.zeros(world.dim_p)

            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.state.p_ang = 2 * np.pi * np.random.uniform(0, 1)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def distance_between_agent(self, world):
        agent_position = np.array([])
        dist = 0
        for a in world.agents:
            agent_position = np.append(agent_position, a.state.p_pos)
        agent_position = agent_position.reshape(-1,2)
        #num_agent = agent_position.shape[0]
        for i in range(world.num_agents):
            current_agent_pos = agent_position[i,:]
            dist += np.sum(np.sqrt(np.sum(np.square(agent_position - current_agent_pos), axis=1)))

        return dist/2

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        alpha = 0.01
        beta1 = 100 # bigbig
        beta2 = 0.001
        beta3 = 1
        beta4 = 1
        rew1, rew2, rew3, rew4, rew5, rew6 = 0, 0, 0, 0, 0, 0
        all_dist = []
        all_ang_dist = []
        for a in world.agents:
            rew1 += np.sum(a.state.p_vel**2)*alpha
        for l, landmark in enumerate(world.landmarks):
            if l < world.num_goals:
                dists_goal = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                rew2 -= min(dists_goal)*beta1
                all_dist = all_dist + [min(dists_goal)]
            if l >=world.num_goals: # agent distance to obstacle
                dists_ob = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                rew3 += min(dists_ob)*beta2
        if world.num_agents>1:
            rew4 = self.distance_between_agent(world)*beta2

        ang_dist = []
        for i, landmark in enumerate(world.landmarks):
            if i <world.num_goals:
                # cos_dist = [np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos))) for a in world.agents]
                diff_angle = abs(agent.state.p_ang-landmark.state.p_ang)
                ang_dist.append(np.pi - abs(diff_angle-np.pi))
        rew5 -= min(ang_dist) * beta3

        #print("rew1",rew1)
        #print("rew2",rew2)
        #print("rew3+rew4",rew3+rew4)
        if agent.collide:
           for a in world.agents:
               if self.is_collision(a, agent):
                   rew6 -= 1 *beta4

        return rew1+rew2+rew3+rew4+rew5+rew6

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
                                                                   
