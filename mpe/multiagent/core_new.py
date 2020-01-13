import numpy as np
import math

def unit_vector(vector):
    #Returns the unit vector of the vector. 
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    #Returns the angle in radians between vectors 'v1' and 'v2'
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        #self.p_vel = None
        # physical angle
        self.p_ang = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # physical rotated angle
        #self.phi = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # length between wheels (0.6*self.size)
        self.length = 0.030
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 0.8#1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()

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
        self.u_range = 1.0
        # control phi range
        self.phi_range = 30.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # angle dimensionality
        self.dim_ang = 1
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        #np.random.uniform(-1, 1) * 30 
        #entity.action.phi = phi = apply_rotation_angle(self)    
        # gather forces applied to entities
        #p_force = [None] * len(self.entities)
        # apply agent physical controls
        #p_force = self.apply_action_force(p_force)
        #print('force:',p_force)
        # apply environment forces
        #p_force = self.apply_environment_force(p_force)
        self.constraint_action_within_range()
        # generate real action by adding noise
        #self.add_noise_in_action()
        # integrate physical state
        self.integrate_agent_state()
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
    
    # gather agent rotation angle
    # def apply_rotation_angle(self):
    #     # generate any angle betweem -10 and 10
    #     phi = np.random.uniform(-1, 1) * 30
    #     return phi

    # gather agent action forces
    # def apply_action_force(self, p_force):
    #     # set applied forces
    #     for i,agent in enumerate(self.agents):
    #         if agent.movable:
    #             noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
    #             #print("agent.action.u",agent.action.u)
    #             #print("noise:",noise)
    #             p_force[i] = agent.action.u + noise                
    #     return p_force

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
    # def apply_environment_force(self, p_force):
    #     # simple (but inefficient) collision response
    #     for a,entity_a in enumerate(self.entities):
    #         for b,entity_b in enumerate(self.entities):
    #             if(b <= a): continue
    #             [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
    #             if(f_a is not None):
    #                 if(p_force[a] is None): p_force[a] = 0.0
    #                 p_force[a] = f_a + p_force[a] 
    #             if(f_b is not None):
    #                 if(p_force[b] is None): p_force[b] = 0.0
    #                 p_force[b] = f_b + p_force[b]        
    #     return p_force

    # integrate physical state

    def integrate_agent_state(self):
        print("****")
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)   
            if entity.action.u is not None:
                if entity.action.u[0]>0:
                    entity.state.p_pos[0] += entity.action.u[0] * self.dt * np.cos(entity.state.p_ang)
                    entity.state.p_pos[1] += entity.action.u[0] * self.dt * np.sin(entity.state.p_ang)
                    entity.state.p_ang += entity.action.u[1] * self.dt
                elif entity.action.u[0]<0:
                    entity.state.p_ang = self.contraint_angle_range(entity.state.p_ang+np.pi)
                    entity.state.p_pos[0] -= entity.action.u[0] * self.dt * np.cos(entity.state.p_ang)
                    entity.state.p_pos[1] -= entity.action.u[0] * self.dt * np.sin(entity.state.p_ang)
                    entity.state.p_ang += entity.action.u[1] * self.dt   
                else:
                    entity.state.p_ang += entity.action.u[1] * self.dt                     
            entity.state.p_ang = self.contraint_angle_range(entity.state.p_ang)        
            # if entity.state.p_ang <= 0:
            #     entity.state.p_ang += 2 * math.pi
            # if entity.state.p_ang >= 2 * math.pi:
            #     entity.state.p_ang -= 2 * math.pi
    def contraint_angle_range(self, theta):
        if theta <= 0:
            theta += 2*np.pi
        if theta >= 2*np.pi:
            theta -= 2*np.pi 
        return theta    
    # integrate physical state
    # def integrate_state(self, p_force):
    #     for i,entity in enumerate(self.entities):
    #         if not entity.movable: continue
    #         #entity.action.vel = entity.action.vel * (1 - self.damping)
    #         entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
    #         if (p_force[i] is not None):
    #             #entity.action.vel += (p_force[i] / entity.mass) * self.dt
    #             p_force_total = np.sqrt(p_force[i][0]**2 + p_force[i][1]**2)
    #             p_force[i][0], p_force[i][1] = p_force_total*np.cos(entity.state.p_ang/180*np.pi), p_force_total*np.sin(entity.state.p_ang/180*np.pi)
    #             entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
    #         #speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
    #         #entity.state.p_vel[0], entity.state.p_vel[1] = speed*np.cos(entity.state.p_ang/180*np.pi), speed*np.sin(entity.state.p_ang/180*np.pi)       
    #         if entity.max_speed is not None:
    #             #speed = np.sqrt(np.square(entity.action.vel[0]) + np.square(entity.action.vel[1]))
    #             speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
    #             if speed > entity.max_speed:
    #                 entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
    #                                                               np.square(entity.state.p_vel[1])) * entity.max_speed
    #         #entity.state.p_pos += entity.action.vel * self.dt
    #         entity.state.p_pos += entity.state.p_vel * self.dt
    #         entity.state.p_ang += np.sqrt(entity.state.p_vel[0]**2 + entity.state.p_vel[1]**2) * np.tan(entity.action.phi/180*np.pi) * self.dt / entity.length
    #         if entity.state.p_ang <= 0:
    #             entity.state.p_ang += 2 * math.pi
    #         if entity.state.p_ang >= 2 * math.pi:
    #             entity.state.p_ang -= 2 * math.pi

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    # def get_collision_force(self, entity_a, entity_b):
    #     if (not entity_a.collide) or (not entity_b.collide):
    #         return [None, None] # not a collider
    #     if (entity_a is entity_b):
    #         return [None, None] # don't collide against itself
    #     # compute actual distance between entities
    #     delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
    #     dist = np.sqrt(np.sum(np.square(delta_pos)))
    #     # minimum allowable distance
    #     dist_min = entity_a.size + entity_b.size
    #     # softmax penetration
    #     k = self.contact_margin
    #     penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
    #     force = self.contact_force * delta_pos / dist * penetration
    #     force_a = +force if entity_a.movable else None
    #     force_b = -force if entity_b.movable else None
    #     return [force_a, force_b]