
# coding: utf-8

# In[1]:


import simpy as sp


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
                return self.lines.request
        else:
            raise self.NoLinesAvailable()
        
    class NoLinesAvailable(sp.exceptions.SimPyException):
        pass


# In[3]:


class Client(object):
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.action = env.process(self.run())
    
    def run(self):
        print(self.name, 'ran at {}'.format(self.env.now))
        try:
            with self.env.call_center.request_line(1)() as req:
                yield req
                print('{} got line at {}'.format(self.name, self.env.now))
                yield self.env.timeout(3)
        except self.env.call_center.NoLinesAvailable as e:
            lines = self.env.call_center.lines
            print('No lines available for {} at {}'.format(self.name, self.env.now), lines.capacity, lines.count)


# In[4]:


env = sp.Environment()
env.call_center = CallCenter(env, 2,0)
clients = [Client(env, i) for i in range(5)]


# In[5]:


env.run(until=10)

