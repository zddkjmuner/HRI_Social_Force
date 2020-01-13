import numpy as np
import math
import random
# mport cvxpy
import matplotlib.pyplot as plt
# import tools
eps = 1e-5

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical angle
        self.p_ang = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None


# state of human model:
class HumanState(EntityState):
    def __init__(self):
        super(HumanState, self).__init__()
        self.c = None

# state of goal:
class GoalState(EntityState):
    def __init__(self):
        super(GoalState, self).__init__()
        self.c = None


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # length between wheels (0.6*self.size)
        self.length = 0.05
        self.movable = False
        # entity collides with others
        self.collide = True
        self.color = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        self.mass = 1.0

# properties of Goal:

class Goal(Entity):
    def __init__(self):
        super(Goal, self).__init__()
        self.movable = True
        self.silent = False
        self.blind = False
        self.u_noise = None
        self.phi_noise = None
        self.c_noise = None
        self.phi_range = 60.0
        self.state = GoalState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        self.size = 0.05
        self.color = np.array([0.35, 0.15, 0.85])

# properties of landmark entities (Human Social force)

class Human(Entity):
    def __init__(self):
        super(Human, self).__init__()
        self.movable = True
        self.silent = False
        self.blind = False
        self.u_noise = None
        self.phi_noise = None
        self.c_noise = None
        self.phi_range = 60.0
        self.state = HumanState()
        self.action = Action()
        self.action_callback = None

        self.eps = 1e-5
        self.v0 = 1.34
        self.vh_max = 1.3 * self.v0
        self.vr_max = 2.0
        self.V0 = 10.0
        self.sigma = 2.0
        self.tau = 0.5
        self.dt = 0.2
        self.phi = 10.0 * np.pi / 9.0
        self.c = 0.5
        self.width = 0.3
        self.color = np.array([0.25, 0.25, 0.25])
        self.mass = 0.6
        self.radius = 0.35 #1.6 
        self.size = 0.05
        self.posX = None
        self.posY = None
        # self.pos = np.random.uniform(-1, +1, 2)
        self.pos = None
        self.dest = np.array([0.0, 0.0])
        # self.velo = None
        self.p_vel = None

class LandmarkState():
    def __init__(self):
        super(LandmarkState, self).__init__()
        self.c = None



class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        self.movable = True
        self.silent = False
        self.blind = False
        self.u_noise = None
        self.phi_noise = None
        self.c_noise = None
        self.phi_range = 60.0
        # state
        self.state = LandmarkState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # physical rotated angle noise amount
        self.phi_noise = None
        # communication noise amount
        self.c_noise = None
        # control u range
        self.u_range = 3.0
        self.phi_range = 20.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

        self.eps = 1e-5
        self.v0 = 1.34
        self.vh_max = 1.3 * self.v0
        self.vr_max = 2.0
        self.V0 = 10.0
        self.sigma = 2.0
        self.tau = 0.5
        self.dt = 0.2
        self.phi = 10.0 * np.pi / 9.0
        self.c = 0.5
        self.width = 0.3
        self.size = 0.05
        self.color = np.array([0.35, 0.35, 0.85])

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        # self.landmarks = [] # split landmarks to human model and goal
        self.humans = []
        self.goals = []

        self.eps = 1e-5
        # self.eps = 1e-5
        self.v0 = 1.34
        self.vh_max = 1.3 * self.v0
        self.vr_max = 2.0
        self.V0 = 10.0
        self.sigma = 2.0
        self.tau = 0.5
        # self.dt = 0.2
        self.dt = 0.05
        self.phi = 10.0 * np.pi / 9.0
        self.c = 0.5
        # communication channel dimensionality
        self.dim_c = 0
        # # position dimensionality
        self.dim_p = 2

    # return all entities in the world
    @property
    def entities(self):
        # return self.agents + self.landmarks
        return self.agents + self.goals + self.humans

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts

    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    @property
    def scripted_goals(self):
        return [goal for goal in self.goals if goal.action_callback is not None]
    
    @property
    def scripted_humans(self):
        return [human for human in self.humans if human.action_callback is not None]

    def _norm(self, x):
        return x / (np.linalg.norm(x) + self.eps)

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        
        # set actions for scripted goals:
        for goal in self.scripted_goals:
            goal.action = goal.action_callback(goal, self)

        # set actions for scripted humans:
        for human in self.scripted_humans:
            human.action = human.action_callback(human, self)

        # integrate physical state

        self.integrate_agent_state()
        # self.integrade_goal_state()

        self.integrade_human_state()

        # self.integrate_landmark_state()
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        
    # gather agent acceleration
    def gather_agent_acceleration(self):
        for i, agent in enumerate(self.agents):
            if agent.movable:
                agent.state.acc = agent.action.u[0]
                agent.state.ag_acc = agent.action.u[1]

    def gather_human_acceleration(self):
        for i, human in enumerate(self.humans):
            human.state.acc = human.action.u[0]
            human.state.ag_acc = human.action.u[1]
            print("!!!!!!!!!!!")
            print(human.action.u)


    def constraint_action_within_range(self):
        for entity in self.entities:
            if not entity.movable: continue
            entity.action.u[0] = 2.5 * entity.action.u[0]    
            if entity.action.u[1] > 2.2 and entity.action.u[1] < -2.2:
                entity.action.u[1] = 2.2 * np.random.uniform(-1,1)

    # generate real action by adding noise
    def add_noise_in_action(self):
        for agent in self.agents:
            if agent.movable:
                print("velocity", agent.action.u[0])
                print("angular velocity", agent.action.u[1])
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                agent.action.u += noise

    # gather physical forces acting on entities
    def apply_environment_force(self):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(entity_a.action.u[0] is None): entity_a.action.u[0] = 0.0
                    #print("acc:",entity_a.action.u[0]) 
                    #print("f_a:",f_a)
                    #print("mass:",entity_a.mass)   
                    entity_a.action.u[0] += f_a/entity_a.mass  
                if(f_b is not None):
                    if(entity_b.action.u[0] is None): entity_b.action.u[0] = 0.0
                    entity_b.action.u[0] += f_b/entity_b.mass         

    # integrate physical state

    def integrate_agent_state(self):
        """ for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            if entity.action.u is not None: """

        for i,agent in enumerate(self.agents):
            if not agent.movable: continue
            if agent.action.u is not None:

                # Non holonomic dynamics
                agent.action.u[0] = np.clip(agent.action.u[0], -1., 1.) # acceleration
                agent.action.u[1] = np.clip(agent.action.u[1], -math.pi/3, math.pi/3) # steering angle 

                speed, heading = agent.state.p_vel[0], agent.state.p_vel[1]

                agent.state.p_pos[0] += speed * np.cos(heading) * self.dt
                agent.state.p_pos[1] += speed * np.sin(heading) * self.dt
                agent.state.p_vel[1] += speed / agent.length * np.tan(agent.action.u[1]) * self.dt
                agent.state.p_vel[1] = math.asin(math.sin(agent.state.p_vel[1]))
                agent.state.p_ang = heading
                agent.state.p_vel[0] += agent.action.u[0] * self.dt                

                agent.state.p_pos[0] = np.clip(agent.state.p_pos[0], -1.5, 1.5)
                agent.state.p_pos[1] = np.clip(agent.state.p_pos[1], -1.5, 1.5)
                agent.state.p_vel[0] = np.clip(agent.state.p_vel[0], -2, 2)
                agent.state.p_vel[1] = np.clip(agent.state.p_vel[1], -2*np.pi, 2*np.pi)
    ######################################################################

    def integrade_human_state(self):
        e_h = [self._norm(human.dest - human.pos)  for human in self.humans]
        e_r = []
        for id_r, robot in enumerate(self.agents):
            for id_g, goal in enumerate(self.goals):
                if id_g == id_r:
                    add_e = self._norm(goal.state.p_pos - robot.state.p_pos)
                    e_r.append(add_e)

        F_robot = [np.zeros(shape=2) for _ in range(len(self.agents))]
        F_human = [np.zeros(shape=2) for _ in range(len(self.humans))]
        # roboInter = 0.0
        # peopleInter = 0.0
        for id_h, human in enumerate(self.humans):
            F_human[id_h] += (self.v0 * e_h[id_h] - human.p_vel) / self.tau
            for id_r, robot in enumerate(self.agents):
                r0 = human.pos - robot.state.p_pos
                s = np.linalg.norm(robot.state.p_vel) * self.dt
                n0 = np.linalg.norm(r0)
                n1 = np.linalg.norm(r0 - s * e_r[id_r])
                b = 0.5 * np.sqrt(np.max([np.square(n0 + n1) - np.square(s), 0.0]))
                f = 0.25 * self.V0 * np.exp(- b / self.sigma) * (2.0 + n0 / (n1 + self.eps) 
                    + n1 / (n0 + self.eps)) * r0 / (self.sigma * b + self.eps)
                w = 1.0 if np.dot(e_r[id_r], -f) >= np.linalg.norm(f) * np.cos(self.phi) else self.c
                F_human[id_h] += w * f / 1000

            for id_h_j, human_j in enumerate(self.humans):
                r0 = human.pos - human_j.pos
                s = np.linalg.norm(human_j.p_vel) * self.dt
                n0 = np.linalg.norm(r0)
                n1 = np.linalg.norm(r0 - s * e_h[id_h_j])
                b = 0.5 * np.sqrt(np.max([np.square(n0 + n1) - np.square(s), 0.0]))
                f = 0.25 * self.V0 * np.exp(- b / self.sigma) * (2.0 + n0 / (n1 +
                    self.eps) + n1 / (n0 + self.eps)) * r0 / (self.sigma * b + self.eps)
                w = 1.0 if np.dot(e_h[id_h_j], -f) >= np.linalg.norm(f) * np.cos(self.phi) else self.c
                F_human[id_h] += w * f / 5000
        
        w_h = []
        for id_h, human in enumerate(self.humans):
            add_w_h = human.p_vel + F_human[id_h] * self.dt
            w_h.append(add_w_h)
        
        v_max = [self.vr_max for _ in range(len(self.humans))]
        # v_new = [self._norm(w_h[id_h]) * np.min([np.linalg.norm(w_h[id_h]), v_max[id_h]]) for id_h in range(len(self.humans))]
        
        v_new = []
        # update human's velocity & position:
        for id_h, human in enumerate(self.humans):
            v_new_add = self._norm(w_h[id_h]) * np.min([np.linalg.norm(w_h[id_h]), v_max[id_h]])
            v_new.append(v_new_add)
            # human.p_vel = v_new_add

        for id_h, human in enumerate(self.humans):
            human.p_vel = v_new[id_h]
            # print("v_new:{}".format(v_new[id_h]))
            human.pos = human.pos + v_new[id_h] * self.dt
            cmp_ang_fenzi = human.p_vel[0]
            cmp_ang_fenmu = np.sqrt(np.sum(np.square(human.p_vel)))
            cmp_ang = cmp_ang_fenzi / cmp_ang_fenmu
            ang = np.arccos(cmp_ang)
            if human.p_vel[1] >= 0:
                human.p_ang = ang
            else:
                human.p_ang = - ang + 2 * np.pi


    #######################################################################
    def contraint_angle_range(self, theta):

        if theta <= 0:
            theta += 2*np.pi
        if theta >= 2*np.pi:
            theta -= 2*np.pi 
        return theta    

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    def update_human_state(self, human):
        if not human.silent:
            human.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*human.action.c.shape) * human.c_noise if human.c_noise else 0.0
            human.state.c = human.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):

        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force = np.sqrt(np.sum(np.square(force)))
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None

        return [force_a, force_b]