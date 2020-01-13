import numpy as np
from multiagent.core import World, Agent, Landmark, Goal, Human, Entity
# from multiagent.core import *
# import core
from multiagent.scenario import BaseScenario
# import tools
import random

def worldCoord2ScreenCoord(worldCoord,screenSize, res):
    wc = np.append(worldCoord,1.0)
    # 要翻转y轴
    mirrorMat = np.matrix([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    scaleMat = np.matrix([
        [res,0.0,0.0],
        [0.0,res,0.0],
        [0.0,0.0,1.0]
    ])
    transMat = np.matrix([
        [1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,screenSize[1],1.0]
    ])
    result = wc*scaleMat*mirrorMat*transMat
    return np.array(np.round(result.tolist()[0][:2]),dtype=int)


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm

def g(x):
    return np.max(x, 0)


def distanceP2W(point, wall):
    p0 = np.array([wall[0],wall[1]])
    p1 = np.array([wall[2],wall[3]])
    d = p1-p0
    ymp0 = point-p0
    t = np.dot(d,ymp0)/np.dot(d,d)
    if t <= 0.0:
        dist = np.sqrt(np.dot(ymp0,ymp0))
        cross = p0 + t*d
    elif t >= 1.0:
        ymp1 = point-p1
        dist = np.sqrt(np.dot(ymp1,ymp1))
        cross = p0 + t*d
    else:
        cross = p0 + t*d
        dist = np.linalg.norm(cross-point)
    npw = normalize(cross-point)
    return dist,npw





def unit_vector(vector):
    #Returns the unit vector of the vector. 
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    #Returns the angle in radians between vectors 'v1' and 'v2'
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))




class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents = 1
        world.num_goals = num_goals = 1
        # world.num_obstacles = num_obstacles = 3
        world.num_humans = num_humans = 5
        world.collaborative = True

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05

        # add goals:
        world.goals = [Goal() for i in range(num_goals)]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal %d' % i
            goal.collide = True
            goal.movable = False
            goal.size = 0.05

        # add humans:
        world.humans = [Human() for i in range(num_humans)]
        for i, human in enumerate(world.humans):
            human.name = 'human %d' % i
            human.collide = True
            human.movable = False
            human.size = 0.02
        
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])

        for goal in world.goals:
            goal.color = np.array([0.35, 0.15, 0.05])

        for human in world.humans:
            human.color = np.array([0.25, 0.25, 0.25])

        human_start_posX = random.uniform(- 0.8, + 0.8)
        human_start_posY = random.uniform(- 0.8, + 0.8)

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_ang = 2 * np.pi * np.random.uniform(0, 1)
            agent.state.p_vel = np.array([0., 0.])
            agent.state.c = np.zeros(world.dim_c)


        for i, goal in enumerate(world.goals):
            goal.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            goal.state.p_vel = np.zeros(world.dim_p)
            goal.state.p_ang = 2 * np.pi * np.random.uniform(0,1)

        for i, human in enumerate(world.humans):
            human.posX = human_start_posX + 0.02 * (2.0 * random.uniform(0,1) - 1)
            human.posY = human_start_posY + 0.02 * (2.0 * random.uniform(0,1) - 1)
            human.pos = np.array([human.posX, human.posY])
            human.p_vel = np.zeros(shape=2)
            human.p_ang = 2 * np.pi * np.random.uniform(0,1)

# set world completed
#################################################################################################################



    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_humans = 0
        min_dists = 0
        for h in world.humans:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - h.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_humans += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_humans)


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
        alpha = 0.0
        beta1 = 100 # bigbig
        beta2 = 0.0
        beta3 = 0
        beta4 = 10
        rew1, rew2, rew3, rew4, rew5, rew6 = 0, 0, 0, 0, 0, 0
        all_dist = []
        all_ang_dist = []
        for a in world.agents:
            rew1 += np.sum(a.state.p_vel**2)*alpha

        for g, goal in enumerate(world.goals):
            dists_goal = [np.sqrt(np.sum(np.square(a.state.p_pos - goal.state.p_pos))) for a in world.agents]
            rew2 -= min(dists_goal)*beta1
            all_dist = all_dist + [min(dists_goal)]
        
        for h, human in enumerate(world.humans):
            # dists_ob = [np.sqrt(np.sum(np.square(a.state.p_pos - human.state.p_pos))) for a in world.agents]
            # rew3 += min(dists_ob)*beta2

            dists_ob = [np.sqrt(np.sum(np.square(a.state.p_pos - human.pos))) for a in world.agents]
            rew3 += min(dists_ob)*beta2

        # angle rewards:
        ang_dist = []
        for i, goal in enumerate(world.goals):
            diff_angle = abs(agent.state.p_ang - goal.state.p_ang)
            ang_dist.append(np.pi - abs(diff_angle-np.pi))
            rew5 -= min(ang_dist) * beta3

        if agent.collide:
           for a in world.agents:
               if self.is_collision(a, agent):
                   rew6 -= 1 *beta4

        return 0.6*rew2 + rew6

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for g in world.goals:
            entity_pos.append(g.state.p_pos - agent.state.p_pos)

        for h in world.humans:
            entity_pos.append(h.pos - agent.state.p_pos)

        # entity colors
        entity_color = []
        for g in world.goals:
            entity_color.append(g.color)

        for h in world.humans:
            entity_color.append(h.color)

        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)