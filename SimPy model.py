
# coding: utf-8

# In[1]:


import simpy as sp
import numpy as np


# Simpy documentation - https://simpy.readthedocs.io/en/latest/contents.html 

# # Testing model

# In[2]:


class CallCenter(object):
    def __init__(self, env, n_lines, n_vip_lines):
        self.env = env
        self.n_lines = n_lines
        self.n_vip_lines = n_vip_lines
        self.lines = sp.Resource(env, capacity=self.n_lines)

    def request_line(self, client_priority):
        n_lines_to_give = self.n_lines-self.n_vip_lines if client_priority==3 else self.n_lines
        if self.lines.count<n_lines_to_give:
            req = self.lines.request()
            yield req
        else:
            raise self.NoLinesAvailable()
        yield self.env.timeout(8)
        return req
        
    def release_line(self, req):
        yield self.lines.release(req)
    
    class NoLinesAvailable(sp.exceptions.SimPyException):
        pass


# In[3]:


class Client(object):
    def __init__(self, env, name, priority):
        self.env = env
        self.name = name
        self.priority = priority
        self.action = env.process(self.run())
    
    def run(self):
        #print(self.name, 'ran at {}'.format(self.env.now))
        cc = self.env.call_center
        try:
            print('{} asking for line at {}'.format(self.name, self.env.now))
            req = yield self.env.process(cc.request_line(self.priority))
            print('{} get line at {}'.format(self.name, self.env.now))
        except cc.NoLinesAvailable as e:
            print('No lines available for {} at {}. Occupied lines: {}'.format(self.name, self.env.now, cc.lines.count))
            return
        yield self.env.timeout(3)
        yield self.env.process(cc.release_line(req))


# In[4]:


def client_generator(env):
    cl_number = 0
    while True:
        if np.random.rand()<0.5:
            client = Client(env, cl_number, 3)
            cl_number += 1
        yield env.timeout(1)


# In[5]:


env = sp.Environment()
env.call_center = CallCenter(env, 2,0)
env.client_generator = env.process(client_generator(env))


# In[6]:


env.run(until=50)

