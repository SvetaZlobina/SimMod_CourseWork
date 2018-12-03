
# coding: utf-8

# In[1]:


import simpy


# https://simpy.readthedocs.io/en/latest/contents.html

# # basic

# In[2]:


def car(env):
    while True:
        print('Start parking at %d' % env.now)
        parking_duration = 5
        yield env.timeout(parking_duration)
        
        print('Start driving at %d' % env.now)
        trip_duration = 2
        yield env.timeout(trip_duration)


# In[3]:


env = simpy.Environment()
env.process(car(env))
env.run(until=15)


# # Process interaction

# In[4]:


class Car(object):
    def __init__(self, env):
        self.env = env
        self.action = env.process(self.run())
    
    def run(self):
        while True:
            print('Start parking and charging at %d' % self.env.now)
            charge_duration = 5
            yield self.env.process(self.charge(charge_duration))
            
            print('start driving at %d'%self.env.now)
            trip_duration = 2
            yield self.env.timeout(trip_duration)
    
    def charge(self, duration):
        yield self.env.timeout(duration)


# In[5]:


env = simpy.Environment()
car = Car(env)
env.run(until=15)


# In[6]:


def driver(env, car):
    yield env.timeout(3)
    car.action.interrupt()


# In[7]:


class Car(object):
    def __init__(self, env):
        self.env = env
        self.action = env.process(self.run())
    
    def run(self):
        while True:
            print('Start parking and charging at %d' % self.env.now)
            charge_duration = 5
            try:
                yield self.env.process(self.charge(charge_duration))
            except simpy.Interrupt:
                print('Was interrupted. Hope, the battery is full enough ...')
            
            print('start driving at %d'%self.env.now)
            trip_duration = 2
            yield self.env.timeout(trip_duration)
    
    def charge(self, duration):
        yield self.env.timeout(duration)


# In[8]:


env = simpy.Environment()
car = Car(env)
env.process(driver(env, car))
env.run(until=15)


# # Resources

# In[9]:


def car(env, name, bcs, driving_time, charge_duration):
    yield env.timeout(driving_time)
    
    print('%s arriving at %d' % (name, env.now))
    with bcs.request() as req:
        yield req
        
        print('%s starting to charge at %s' % (name, env.now))
        yield env.timeout(charge_duration)
        print('%s leaving the bcs at %s' % (name, env.now))


# In[10]:


env = simpy.Environment()
bcs = simpy.Resource(env, capacity=2)


# In[11]:


for i in range(4):
    env.process(car(env, 'Car %d'%i, bcs, i*2, 5))


# In[12]:


env.run()


# # Events

# In[13]:


class School:
    def __init__(self, env):
        self.env = env
        self.class_ends = env.event()
        self.pupil_procs = [env.process(self.pupil()) for i in range(3)]
        self.bell_proc = env.process(self.bell())
    
    def bell(self):
        print('start bell')
        for i in range(2):
            print('before bell timeout')
            yield self.env.timeout(45)
            print('before succeed')
            self.class_ends.succeed()
            print('after succeed')
            self.class_ends = self.env.event()
            print()
    def pupil(self):
        print('start pupil', end='')
        for i in range(2):
            print(r' \o/', end='')
            print()
            yield self.class_ends
            print(' /o\\', end='')


# In[14]:


school = School(env)
env.run()


# # Basic resource

# In[15]:


def resource_user(env, resource):
    request = resource.request()
    yield request
    yield env.timeout(1)
    resource.release(request)
    
env = simpy.Environment()
res = simpy.Resource(env, capacity=1)
user = env.process(resource_user(env, res))
env.run()


# In[16]:


res = simpy.Resource(env, capacity=1)
def print_stats(res):
    print('%d of %d slots are allocated.' %(res.count, res.capacity))
    print(' Users: ', res.users)
    print(' Queued events:', res.queue)
    
def user(res):
    print_stats(res)
    with res.request() as req:
        yield req
        print_stats(res)
    print_stats(res)
procs = [env.process(user(res)), env.process(user(res))]


# In[17]:


env.run()


# ## Priority resource

# In[18]:


def resource_user(name, env, resource, wait, prio):
    yield env.timeout(wait)
    with resource.request(priority=prio) as req:
        print('%s requesting at %s with priority=%s' % (name, env.now, prio))
        yield req
        print('%s got resource at %s' % (name, env.now))
        yield env.timeout(3)


# In[19]:


env = simpy.Environment()
res = simpy.PriorityResource(env, capacity=1)
p1 = env.process(resource_user(1, env, res, wait=0, prio=0))
p2 = env.process(resource_user(2, env, res, wait=1, prio=0))
p3 = env.process(resource_user(3, env, res, wait=2, prio=-1))
env.run()

