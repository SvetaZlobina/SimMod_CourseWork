
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import importlib

from tqdm import tqdm_notebook


# In[2]:


try: importlib.reload(msp)
except: import ModSimPy as msp  # Библиотека с github - предназначена для имит. моделирования


# # Helpful functions

# In[3]:


type_priority_mapping = {k:v for k,v in zip(['gold','silver','regular'],range(1,4))}
type_salary_mapping = {k:v for k,v in zip(['gold','silver','regular'], [600*8, 500*8, 400*8])}


# In[4]:


def to_next_timestep(system, state, tqdm):
    """
    Переводит систему в следующий момент времени
    """
    state.time_cur += system.timedelta
    tqdm.update(system.timedelta.seconds)


# In[5]:


def calc_statistic(state):
    """
    Требуемая от симуляции статистика
    """
    data = pd.Series({
        #'busy_lines': get_n_busy_lines(state.lines_time_to_free),
        #'missed_clients_regular':state.n_missed_clients['regular']
        },
        name=state.time_cur)
    return data


# In[6]:


def get_datetime(hours, mins, secs=None):
    return datetime.datetime(2018,1,1,hours,mins, secs if secs else 0)


# # Подготовка данных

# Заданные значения частот звонков разных клиентов

# In[7]:


calls_stat_ds = pd.DataFrame()
calls_stat_ds['regular_clients'] = [87, 165, 236, 323, 277, 440, 269, 342, 175, 273, 115,  56]
calls_stat_ds['vip_clients'] = [89, 243, 221, 180, 301, 490, 394, 347, 240, 269, 145,  69]
calls_stat_ds['silver_clients'] = 0.68*calls_stat_ds['vip_clients']
calls_stat_ds['gold_clients'] = calls_stat_ds['vip_clients']-calls_stat_ds['silver_clients']
for f in calls_stat_ds.columns:
    calls_stat_ds[f+'_per_sec'] = calls_stat_ds[f]/3600
calls_stat_ds.index = range(7,19)
print('Частота звонков')
calls_stat_ds


# In[8]:


operators_ds = pd.DataFrame(columns=['id', 'type', 'start_work_time'],
                            data=[[0, 'regular', get_datetime(7,0)],
                                  [1, 'regular', get_datetime(11,0)]])


# # Модель 1

# In[9]:


def init_state(system):
    """
    Задаёт первичное состояние системы
    """
    state = msp.State(
        time_cur=system.time_start,  # Текущее время
        # Данные по линиям связи (возможно уже не нужны)
        lines_ds=pd.DataFrame({'time_to_free':0 for i in range(system.n_lines)}, index=range(system.n_lines)),
        # Данные по всем клиентам
        clients_ds=pd.DataFrame(columns=['id','type','call_start_time', 'max_waiting_time', 'missed']),
        # Данные по всем соединениям операторов с клиентами
        connections_ds=pd.DataFrame(columns=['id', 'operator_id', 'client_id', 'line_id', 'time_start','time_to_service', 'closed']),
        # Очередь клиентов на соединение
        queue_ds=pd.DataFrame(columns=['client_id','priority', 'time_from']  # 
        )
    )
    return state


# In[10]:


def generate_clients(system, state):
    """
    Генератор клиентов. Использует заданные частоты звонков клиентов.
    За одну секунду генерируется несколько клиентов, т.к. могут позвонить одновременно золотой и обычные клиент.
    """
    probs = [system.calls_stat.loc[state.time_cur.hour, f'{t}_clients_per_sec'] for t in system.client_types]
    bools = [msp.flip(p) for p in probs]  # Перевод вероятности в True/False
    clients = [ctype for ctype, b in zip(system.client_types, bools) if b] 
    ids = []
    for ctype in clients:
        data = {'id':len(state.clients_ds),
                'type':ctype,
                'call_start_time':state.time_cur,
                'max_waiting_time':3600,  # seconds  # временная константа 
                'missed':False}  # повесил-ли клиент трубку
        ids.append(data['id'])
        state.clients_ds.loc[data['id']] = data
    return ids


# In[11]:


def add_clients_to_queue(state, clients_ids):
    """
    Добавление клиентов с заданными id в очередь ожидания
    """
    for cid in clients_ids:
        data = {'client_id':cid,
                'priority': type_priority_mapping[state.clients_ds.loc[cid,'type']],
                'time_from':state.time_cur}
        state.queue_ds = state.queue_ds.append(data, ignore_index=True)


# In[12]:


def drop_clients_from_queue(state):
    """
    Моделирование "бросания трубки" недождавшихся клиентов
    """
    cds = pd.merge(state.queue_ds, state.clients_ds, left_on='client_id',right_on='id')
    
    missed_cids = cds.loc[([x.seconds for x in state.time_cur-cds['time_from']]>cds['max_waiting_time']), 'client_id']
    if len(missed_cids)>0:
        # Убираем клиента из очереди
        state.queue_ds = state.queue_ds.loc[[not x for x in state.queue_ds['client_id'].isin(missed_cids)]]
        # Запись, что клиент бросил трубку
        state.clients_ds.loc[missed_cids, 'missed'] = True


# In[13]:


def occupy_operators(system, state):
    """
    Поиск свободных операторов, линий и клиентов в очереди. Установка соединений
    """
    connections_open = state.connections_ds[state.connections_ds['closed']==False]
    free_operators = system.operators_ds.loc[[not x for x in system.operators_ds['id'].isin(connections_open['operator_id'])]]
    free_operators = free_operators[(free_operators['start_work_time']<state.time_cur)&
                                    (state.time_cur<free_operators['start_work_time']+system.operators_work_duration)]
    free_lines = [x for x in range(system.n_lines) if x not in connections_open['line_id']]
    
    for idx, op_row in free_operators.iterrows():
        if len(free_lines) == 0:
            break
        cds = state.queue_ds
        cds = cds[cds['priority']<=type_priority_mapping[op_row['type']]]
        if len(cds)==0:
            continue
        cds = cds.sort_values(['priority','time_from'], ascending=[True, False])
        cclient, state.queue_ds = cds.iloc[0], cds.iloc[1:]
        cline, free_lines = free_lines[0], free_lines[1:]
        data = {
            'id': len(state.connections_ds),
            'operator_id':op_row['id'],
            'client_id':cclient['client_id'],
            'line_id':cline,
            'time_start':state.time_cur,
            'time_to_service': datetime.timedelta(seconds=600),
            'closed':False,
            }
        state.connections_ds = state.connections_ds.append(data, ignore_index=True)
    return free_operators


# In[14]:


def release_operators(system, state):
    """
    Закрытие соединений, в которых оператор уже всё отработал
    """
    cds = state.connections_ds[state.connections_ds['closed']==False]
    ended_connections = cds[(state.time_cur-cds['time_start'])>cds['time_to_service']]
    state.connections_ds.loc[ended_connections.index, 'closed'] = True
    return ended_connections


# In[15]:


def step(system, state):        
    """
    Один временной шаг системы
    """
    new_clients_id = generate_clients(system, state)
    add_clients_to_queue(state, new_clients_id)
    
    occupy_operators(system, state)
    release_operators(system, state)
    
    drop_clients_from_queue(state)


# In[16]:


def run_simulation(system):
    """
    Внешняя функция для запуска системы
    """
    state = init_state(system)
    
    results_frame = msp.TimeFrame()
    
    # tqdm - библиотека для рисования прогрессбаров
    tqdm = tqdm_notebook(total=(system.time_end-system.time_start).seconds//system.timedelta.seconds)
    while state.time_cur<system.time_end:
        step(system, state)
        results_frame = results_frame.append(calc_statistic(state))
        to_next_timestep(system, state, tqdm)
    tqdm.close()

    return results_frame, state


# In[17]:


system = msp.System(time_start=get_datetime(7,0),
                #time_end=get_datetime(19,0),
                time_end=get_datetime(9,0),
                timedelta = datetime.timedelta(seconds=1),
                n_lines = 50,  # кол-во линий связи
                calls_stat=calls_stat_ds,  # частоты звонков
                time_to_serve=120,  # seconds # временная константа. время обслуживания каждого клиента
                client_types = ['regular'],  # типы клиентов. затем добавятся silver и gold
                operators_ds=operators_ds,
                operators_work_duration = datetime.timedelta(hours=8)
                )


# In[18]:


results, state_final = run_simulation(system)


# In[19]:


state_final.clients_ds[state_final.clients_ds['missed']==False].head()


# In[20]:


plt.title('missed clients')
plt.plot(state_final.clients_ds['call_start_time'], state_final.clients_ds['missed'].cumsum())
plt.xticks(rotation=45)
plt.show()
print(state_final.clients_ds['missed'].value_counts())

