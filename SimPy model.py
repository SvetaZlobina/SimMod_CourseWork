
# coding: utf-8

# In[1]:


import simpy as sp
import numpy as np


# Simpy documentation - https://simpy.readthedocs.io/en/latest/contents.html 

# # Data preparation

# In[2]:


cl_statuses = ['generated', 'ask_for_line', 'get_line', 'no_lines', 'blocked',
               'unblocked', 'drop_on_unblock', 'in_queue', 'drop_from_queue', 'connected',
               'drop_success']
map_cl_status_code = {s:idx for idx,s in enumerate(cl_statuses)}
map_code_cl_status = {v:k for k,v in map_cl_status_code.items()}


# In[3]:


cl_columns = ['id','priority','call_start_time','call_end_time','max_waiting_time','status']
cl_columns_map = {k:idx for idx,k in enumerate(cl_columns)}


# # Testing model

# In[4]:


class CallCenter(object):
    def __init__(self, env, n_lines, n_vip_lines):
        self.env = env
        self.n_lines = n_lines
        self.n_vip_lines = n_vip_lines
        self.lines = sp.Resource(env, capacity=self.n_lines)

    def request_line(self, cl_id):
        cl_priority = self.env.client_mx[cl_id, cl_columns_map['priority']]
        n_lines_to_give = self.n_lines-self.n_vip_lines if cl_priority==3 else self.n_lines
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


# In[5]:


class Client(object):
    def __init__(self, env, id_):
        self.env = env
        self.id_ = id_
        self.action = env.process(self.run())
    
    def run(self):
        cc = self.env.call_center
        try:
            req = yield self.env.process(cc.request_line(self.id_))
        except cc.NoLinesAvailable as e:
            return
        yield self.env.timeout(3)
        yield self.env.process(cc.release_line(req))


# In[6]:


def add_client_to_matrix(matrix, priority, call_start_time):
    data = np.array([-1]*matrix.shape[1])
    id_ = len(matrix)
    data[cl_columns_map['id']] = id_
    data[cl_columns_map['priority']] = priority
    data[cl_columns_map['call_start_time']] = call_start_time
    data[cl_columns_map['max_waiting_time']] = 5*60
    data[cl_columns_map['status']] = map_cl_status_code['generated']
    return id_, np.append(matrix, [data], axis=0)


# In[7]:


def client_generator(env):
    while True:
        if np.random.rand()<0.5:
            id_, env.client_mx = add_client_to_matrix(env.client_mx, 3, env.now)
            client = Client(env, id_)
        yield env.timeout(1)


# In[8]:


env = sp.Environment()
env.client_mx = np.empty([0,len(cl_columns)])
env.call_center = CallCenter(env, 2,0)
env.client_generator = env.process(client_generator(env))


# In[9]:


env.run(until=50)


# In[10]:


env.client_mx

