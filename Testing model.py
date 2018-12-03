
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    state['time_cur'] += system['timedelta']
    tqdm.update(system['timedelta'].seconds)


# In[5]:


def calc_statistic(state):
    """
    Требуемая от симуляции статистика
    """
    data = pd.Series({        },
        name=state['time_cur'])
    return data


# In[6]:


def get_datetime(hours, mins, secs=None):
    return datetime.datetime(2018,1,1,hours,mins, secs if secs else 0)


# In[7]:


def get_panel(system, state):
    ds = pd.merge(state['connections_ds'], state['clients_ds'],
              left_on='client_id', right_on='id', suffixes=['_con','_client'], how='right').drop('client_id',axis=1)
    ds = pd.merge(ds, system['operators_ds'],
                  left_on='operator_id', right_on='id', suffixes=['','_operator'], how='left').drop('id',axis=1)
    ds = pd.merge(ds, state['queue_ds'],
                  left_on='id_client', right_on='client_id', suffixes=['','_queue'], how='left').drop('client_id',axis=1)
    ds = pd.merge(ds, state['blocked_ds'],
                  left_on='id_client', right_on='client_id', suffixes=['','_block'], how='left').drop('client_id',axis=1)
    ds = ds.rename(columns={'id_con':'id_connection', 'operator_id':'id_operator','line_id':'id_line',
                            'closed':'connection_closed',
                            'type':'type_client', 'opeartor_type':'type_operator',
                            'time_start':'time_start_connection',
                            'call_start_time':'time_start_call',
                            'call_end_time': 'time_end_call',
                            'max_waiting_time':'client_max_waiting_time',
                            'missed':'client_missed',
                            'start_work_time':'operator_start_work_time',
                            'priority':'client_priority',
                            'time_from':'time_queue_from',
                            'exit':'queue_exit',
                            'duration':'block_duration'
                           })
    ds['hour'] = [x.hour for x in ds['time_start_call']]
    ds['client_missed'] = ds['client_missed'].astype(int)
    return ds


# In[37]:


def get_line_loads(panel_ds, time_from=get_datetime(7,0), time_to=get_datetime(19,0)):
    all_time = pd.DataFrame(pd.date_range(time_from, time_to, freq='s'), columns=['ctime'])[:-1]
    all_time.index = all_time['ctime']
    tts = []
    for idx, row in panel_ds.iterrows():
        tt = all_time.loc[row['time_start_call']:row['time_end_call']]
        for f in ['id_connection', 'id_line', 'id_client', 'id_operator']:
            tt[f] = row[f]
        tts.append(tt)
    line_loads_ds = pd.concat(tts)
    del tts
    line_loads_ds = pd.merge(all_time, line_loads_ds, on='ctime', how='left')
    line_loads_ds = line_loads_ds.fillna(-1)
    line_loads_ds = line_loads_ds.pivot_table(index='ctime', columns='id_line', values='id_client')
    if -1 in line_loads_ds.columns: line_loads_ds = line_loads_ds.drop(-1,axis=1)
    #line_loads_ds = line_loads_ds.fillna(-1)
    return line_loads_ds


# In[38]:


def estimate_time_to_wait(system, state, cl_id):
    cl = state['clients_ds'].loc[cl_id]
    mean_time_in_work = {k:datetime.timedelta(seconds=system['time_to_serve']) for k in ['regular', 'silver','gold']}
    
    queue_clients = state['waiting_clients'][cl['type']]
    queue_clients = len(queue_clients)
    available_ops = system['operators_ds'][system['operators_ds']['type']==cl['type']]
    available_ops = available_ops[(available_ops['start_work_time']<=state['time_cur'])&
                                 (state['time_cur']<=available_ops['start_work_time']+system['operators_work_duration'])]
    available_ops = len(available_ops)
    
    if available_ops==0:
        time_to_wait = np.inf
    else:
        time_to_wait = (queue_clients*mean_time_in_work[cl['type']]).seconds/available_ops
    return time_to_wait


# In[10]:


def add_operator(type_, start_work_time, ds):
    ds.loc[len(ds)] = {'id':len(operators_ds), 'type':type_,
                       'priority':type_priority_mapping[type_], 'start_work_time':start_work_time}
def add_operators(ar, ds):
    for t, swt in ar: add_operator(t, swt, ds)


# # Подготовка данных

# Заданные значения частот звонков разных клиентов

# In[11]:


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


# In[12]:


operators_ds = pd.DataFrame(columns=['id', 'type', 'priority','start_work_time'])
for f in ['id','priority']:
    operators_ds[f] = operators_ds[f].astype(int)

add_operators([['regular', get_datetime(7,0)],
               ['regular', get_datetime(11,0)],
               ['silver',  get_datetime(7,0)],
               ['gold',    get_datetime(8,0)]],
             operators_ds)

for i in range(50):
    add_operator('regular', get_datetime(7,0), operators_ds)
# # Модель 1

# In[13]:


def init_state(system):
    """
    Задаёт первичное состояние системы
    """
    state = {
        'time_cur': system['time_start'],  # Текущее время
        # Данные по всем клиентам
        'clients_ds': pd.DataFrame(columns=['id','line_id','type','call_start_time', 'call_end_time', 'max_waiting_time', 'missed', 'status']),
        # Данные по всем соединениям операторов с клиентами
        'connections_ds': pd.DataFrame(columns=['id', 'operator_id', 'client_id', 'time_start','time_to_service', 'closed']),
        # Очередь клиентов на соединение
        'queue_ds': pd.DataFrame(columns=['client_id', 'priority', 'time_from', 'blocked', 'exit']),  # 
        # Клиенты, которые в очереди, но ждут оценки времени, или вводят номера карт
        'blocked_ds': pd.DataFrame(columns=['client_id','type','time_from','duration', 'unblocked'])
    }
    state['free_lines'] = list(range(system['n_lines']))
    state['free_operators'] =  {t:[] for t in ['regular','silver','gold']}
    state['blocked_clients'] = {t:[] for t in ['regular','silver','gold']}
    state['waiting_clients'] = {t:[] for t in ['regular','silver','gold']}
    
    for ds, f in [['clients_ds', 'id'],
                  ['clients_ds', 'max_waiting_time'],
                  ['clients_ds', 'line_id'],
                  ['connections_ds', 'id'],
                  ['connections_ds', 'operator_id'],
                  ['connections_ds', 'client_id'],
                  ['queue_ds', 'client_id'],
                  ['queue_ds', 'priority'],
                  ['blocked_ds','client_id']]:
        state[ds][f] = state[ds][f].astype(int)
    for ds, f in [['clients_ds', 'missed'],
                  ['connections_ds', 'closed'],
                  ['queue_ds', 'blocked'],
                  ['queue_ds', 'exit'],
                  ['blocked_ds', 'unblocked']]:
        state[ds][f] = state[ds][f].astype(bool)
    return state


# In[14]:


def generate_clients(system, state):
    """
    Генератор клиентов. Использует заданные частоты звонков клиентов.
    За одну секунду генерируется несколько клиентов, т.к. могут позвонить одновременно золотой и обычные клиент.
    """
    probs = [system['calls_stat'].loc[state['time_cur'].hour, f'{t}_clients_per_sec'] for t in system['client_types']]
    bools = [msp.flip(p) for p in probs]  # Перевод вероятности в True/False
    clients = [ctype for ctype, b in zip(system['client_types'], bools) if b] 
    for ctype in clients:
        n_lines_available = len(state['free_lines']) if ctype!='regular' else len(state['free_lines'])-system['n_lines_vip']
        state['free_lines'].sort()
        line_id = state['free_lines'].pop(0) if n_lines_available>0 else None
        data = {'id':len(state['clients_ds']),
                'line_id': line_id,
                'type':ctype,
                'call_start_time':state['time_cur'],
                'call_end_time': None,
                'max_waiting_time':300,  # seconds  # временная константа 
                'missed':False,
                'status':'Generated'}  # повесил-ли клиент трубку
        if line_id is None:
            data['missed'] = True
            data['call_end_time'] = state['time_cur']
            data['status'] = 'No_lines_available'
        state['clients_ds'].loc[data['id']] = data


# In[15]:


def add_clients_to_queue(state):
    """
    Добавление клиентов с заданными id в очередь ожидания
    """
    new_clients = state['clients_ds'][state['clients_ds']['call_start_time']==state['time_cur']]
    new_clients = new_clients[new_clients['missed']==False]
    for idx, row in new_clients.iterrows():
        data = {'client_id': row['id'],
                'priority': type_priority_mapping[row['type']],
                'time_from': state['time_cur'],
                'blocked': False,
                'exit': False}
        state['queue_ds'].loc[row['id']] = data
        state['waiting_clients'][row['type']].append(row['id'])
    state['clients_ds'].loc[new_clients['id'], 'status'] = 'Add_to_queue'


# In[16]:


def drop_clients_from_queue(state):
    """
    Моделирование "бросания трубки" недождавшихся клиентов
    """
    cds = pd.merge(state['queue_ds'], state['clients_ds'], left_on='client_id',right_on='id')
    
    cds = cds[cds['exit']==False]
    cds = cds[cds['blocked']==False]
    #missed = cds.loc[([x.seconds for x in state['time_cur']-cds['time_from']]>cds['max_waiting_time'])]
    missed = cds.loc[state['time_cur']-cds['time_from']>system['very_long_waiting']]
    if len(missed)>0:
        state['queue_ds'].loc[missed['client_id'], 'exit'] = True
        # Запись, что клиент бросил трубку
        state['clients_ds'].loc[missed['client_id'], 'missed'] = True
        state['clients_ds'].loc[missed['client_id'], 'call_end_time'] = state['time_cur']
        # Удаление клиента из очереди, оптимизирующей расчёты
        for idx, row in missed.iterrows():
            for ds in ['waiting_clients', 'blocked_clients']:
                if row['client_id'] in state[ds][row['type']]:
                    state[ds][row['type']].remove(row['client_id'])
            state['free_lines'].append(row['line_id'])
        state['clients_ds'].loc[missed['client_id'], 'status'] = 'Dropped_from_queue'


# In[17]:


def block_clients_in_queue_regular(state):
    cds = state['queue_ds'][state['queue_ds']['priority']==3]
    cds = cds[cds['exit']==False]
    cds = cds[cds['blocked']==False]
    cds = cds[(state['time_cur']-cds['time_from'])==datetime.timedelta(seconds=1)]
    for i in cds['client_id']:
        is_missed = msp.flip(0.1)
        if is_missed:
            state['queue_ds'].at[i, 'exit'] = True
            state['clients_ds'].at[i, 'missed'] = True
            state['clients_ds'].at[i, 'call_end_time'] = state['time_cur']
            for q in ['waiting_clients', 'blocked_clients']:
                if i in state[q]['regular']:
                    state[q]['regular'].remove(i)
            state['free_lines'].append(state['clients_ds'].at[i,'line_id'])
            state['clients_ds'].at[i, 'status'] = 'Dropped_on_block'
        else:
            data = {'client_id':i,
                'type':'regular',
                'time_from': state['time_cur'],
                'duration': datetime.timedelta(seconds=7),
                'unblocked': False}
            state['queue_ds'].at[cds['client_id'], 'blocked'] = True
            state['blocked_ds'].loc[len(state['blocked_ds'])] = data
            if i in state['waiting_clients']['regular']:
                state['blocked_clients']['regular'].append(i)
                state['waiting_clients']['regular'].remove(i)
            state['clients_ds'].at[i, 'status'] = "Blocked"


# In[18]:


def block_clients_in_queue_vip(state):
    cds = state['queue_ds'][state['queue_ds']['priority']<3]
    cds = cds[cds['exit']==False]
    cds = cds[cds['blocked']==False]
    cds = cds[cds['time_from']==state['time_cur']]
    state['queue_ds'].loc[cds['client_id'], 'blocked'] = True
    for i in cds['client_id']:
        t = state['clients_ds'].at[i, 'type']
        data = {'client_id':i,
                'type':t,
                'time_from': state['time_cur'],
                'duration': datetime.timedelta(seconds=10), #TEMP constant
                'unblocked': False
               } 
        state['blocked_ds'].loc[len(state['blocked_ds'])] = data
        if i in state['waiting_clients'][t]:
            state['waiting_clients'][t].remove(i)
        state['blocked_clients'][t].append(i)
    state['clients_ds'].loc[cds['client_id'], 'status'] = 'Blocked'


# In[19]:


def block_clients_in_queue(state):  
    block_clients_in_queue_regular(state)
    block_clients_in_queue_vip(state)

def unblock_clients_in_queue(state):
    cds = state['blocked_ds']
    cds = cds[cds['unblocked']==False]
    cds = cds[state['time_cur']>cds['time_from']+cds['duration']]
    
    state['blocked_ds'].loc[cds.index, 'unblocked'] = True
    state['queue_ds'].loc[cds['client_id'], 'blocked'] = False
    items = state['blocked_clients'].items()
    for t, blocked_ids in items:
        for i in blocked_ids:
            time_to_wait = estimate_time_to_wait(system, state, i)
            if time_to_wait<=state['clients_ds'].at[i,'max_waiting_time']:
                state['blocked_clients'][t].remove(i)
                state['waiting_clients'][t].append(i)
                state['clients_ds'].at[i, 'status'] = 'Unblocked'
            else:
                state['queue_ds'].at[i, 'exit'] = True
                state['clients_ds'].at[i, 'missed'] = True
                state['clients_ds'].at[i, 'call_end_time'] = state['time_cur']
                for q in ['waiting_clients', 'blocked_clients']:
                    if i in state[q][t]:
                        state[q][t].remove(i)
                state['free_lines'].append(state['clients_ds'].at[i,'line_id'])
                state['clients_ds'].at[i, 'status'] = 'Dropped_on_unblock'
# In[20]:


def unblock_clients_in_queue(state):
    cds = state['blocked_ds']
    cds = cds[cds['unblocked']==False]
    cds = cds[state['time_cur']>cds['time_from']+cds['duration']]
    
    state['blocked_ds'].loc[cds.index, 'unblocked'] = True
    state['queue_ds'].loc[cds['client_id'], 'blocked'] = False
    for idx, row in cds.iterrows():
        i = row['client_id']
        t = state['clients_ds'].at[i, 'type']
        time_to_wait = estimate_time_to_wait(system, state, i)
        if time_to_wait<=state['clients_ds'].at[i,'max_waiting_time']:
            state['blocked_clients'][t].remove(i)
            state['waiting_clients'][t].append(i)
            state['clients_ds'].at[i, 'status'] = 'Unblocked'
        else:
            state['queue_ds'].at[i, 'exit'] = True
            state['clients_ds'].at[i, 'missed'] = True
            state['clients_ds'].at[i, 'call_end_time'] = state['time_cur']
            for q in ['waiting_clients', 'blocked_clients']:
                if i in state[q][t]:
                    state[q][t].remove(i)
            state['free_lines'].append(state['clients_ds'].at[i,'line_id'])
            state['clients_ds'].at[i, 'status'] = 'Dropped_on_unblock'


# In[21]:


def generate_operators(system, state):
    if state['time_cur'].minute==0 and state['time_cur'].second==0:
        new_operators = system['operators_ds'][system['operators_ds']['start_work_time']==state['time_cur']]
        for idx, row in new_operators.iterrows():
            state['free_operators'][row['type']].append(row['id'])


# In[22]:


def drop_operators(system, state):
    for t, ids in state['free_operators'].items():
        operators = system['operators_ds'].loc[ids]
        operators = operators[operators['start_work_time']+system['operators_work_duration']<=state['time_cur']]
        for i in operators['id']:
            state['free_operators'][t].remove(i)


# In[23]:


def occupy_operators(system, state):
    """
    Поиск свободных операторов, линий и клиентов в очереди. Установка соединений
    """
    for type_op in ['regular','silver','gold']:
        ids_op = state['free_operators'][type_op]
        if len(ids_op)==0: continue 
        clients_available = []
        for t in ['regular', 'silver', 'gold']:
            if type_priority_mapping[type_op] <= type_priority_mapping[t]:
                clients_available = state['waiting_clients'][t]+clients_available
        if len(clients_available)==0: continue
            
        op_cl_pairs = list(zip(ids_op, clients_available))
        for op_id, cl_id in op_cl_pairs:
            for t in ['regular','silver','gold']:
                if cl_id in state['waiting_clients'][t]:
                    state['waiting_clients'][t].pop(0)
            state['free_operators'][type_op].pop(0)
            data = {
                'id': len(state['connections_ds']),
                'operator_id':op_id,
                'client_id':cl_id,
                'time_start':state['time_cur'],
                'time_to_service': datetime.timedelta(seconds=600),
                'closed':False,
                }
            state['connections_ds'] = state['connections_ds'].append(data, ignore_index=True)
            state['queue_ds'].at[cl_id,'exit'] = True 
            state['clients_ds'].at[cl_id, 'status'] = 'Connected'
    return


# In[24]:


def release_operators(system, state):
    """
    Закрытие соединений, в которых оператор уже всё отработал
    """
    cds = state['connections_ds'][state['connections_ds']['closed']==False]
    ended_connections = cds[(state['time_cur']-cds['time_start'])>cds['time_to_service']]
    if len(ended_connections)>0:
        state['connections_ds'].loc[ended_connections.index, 'closed'] = True
        state['clients_ds'].loc[ended_connections['client_id'], 'call_end_time'] = state['time_cur']
        for idx, row in ended_connections.iterrows():
            state['free_operators'][system['operators_ds'].at[row['operator_id'],'type']].append(row['operator_id'])
            state['free_lines'].append(state['clients_ds'].at[row['client_id'],'line_id'])
        state['free_lines'] = sorted(state['free_lines'])
        state['clients_ds'].loc[ended_connections['client_id'], 'status'] = 'Disconnected_success'
    return ended_connections


# In[25]:


def step(system, state):        
    """
    Один временной шаг системы
    """
    generate_clients(system, state)ффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффффф
    add_clients_to_queue(state)
    block_clients_in_queue(state)
    unblock_clients_in_queue(state)
    
    #print('{regular} {silver} {gold}'.format(**{k:state.free_operators[k] for k in['regular','silver','gold']}))
    generate_operators(system, state)
    drop_operators(system, state)   
    
    occupy_operators(system, state)
    release_operators(system, state)
    
    drop_clients_from_queue(state)


# In[26]:


from numba import jit


# In[27]:


def run_simulation(system):
    """
    Внешняя функция для запуска системы
    """
    state = init_state(system)
    
    results_frame = msp.TimeFrame()
    
    # tqdm - библиотека для рисования прогрессбаров
    tqdm = tqdm_notebook(total=(system['time_end']-system['time_start']).seconds//system['timedelta'].seconds)
    while state['time_cur']<system['time_end']:
        """
        try:
            step(system, state)
        except Exception as e:
            print(e)
            return results_frame, state
        """
        step(system,state)
        #results_frame = results_frame.append(calc_statistic(state))
        to_next_timestep(system, state, tqdm)
    tqdm.close()

    return results_frame, state


# In[28]:


system =  {'time_start': get_datetime(7,0),
            #'time_end': get_datetime(19,0),
            'time_end': get_datetime(8,0),
            'timedelta': datetime.timedelta(seconds=1),
            'n_lines': 50,  # кол-во линий связи
            'calls_stat': calls_stat_ds,  # частоты звонков
            'time_to_serve': 120,  # seconds # временная константа. время обслуживания каждого клиента
            'client_types': ['regular', 'silver', 'gold'],  # типы клиентов. затем добавятся silver и gold
            'operators_ds': operators_ds,
            'operators_work_duration': datetime.timedelta(hours=8),
            'n_lines_vip': 5,
            'very_long_waiting': datetime.timedelta(minutes=30)
         }


# In[29]:


results, state_final = run_simulation(system)


# In[30]:


ds = get_panel(system, state_final)
ds.head()


# In[31]:


tt = ds.pivot_table(columns=['hour'], index=['type_client'], values='client_missed', aggfunc='mean').reindex(
    ['gold','silver','regular'])
sns.heatmap(tt, vmin=0, vmax=1, cmap='Reds')
del tt
plt.show()


# In[32]:


line_loads_ds = get_line_loads(ds, time_to=get_datetime(8,0))


# In[33]:


ds[(ds['type_client']=='silver')&(ds['status']!='Dropped_on_unblock')][[
    'time_start_connection','id_client','type_client','time_start_call','time_end_call',
    'status', 'client_max_waiting_time', 'time_from_block','block_duration'
]]


# In[34]:


(line_loads_ds>0).sum(1).fillna(0).plot()
plt.show()


# In[35]:


sns.heatmap(ds.pivot_table(index='type_operator', columns='type_client', values='id_connection', aggfunc='count', fill_value=0),
           cmap='Blues')
plt.show()


# # Что ещё надо добавить

# Разные типы звонков

# Случайный начальные задержки для каждого звонка

# Случайное время готовности ожидания клиента

# Случайное время обслуживания для оператора (+поправки на тип оператора) для соединения

# Поиск оптимальной комбинации
