
import numpy as np
import pandas as pd
import random




# Define the incoming nodes
def generate_dic_node_upstream(network):
    dic_nodes_incoming = {}
    for node in set(network['node2']):
        dic_nodes_incoming[node] = {}
        cur_df = network[network['node2'] == node]
        for node_income in cur_df['node1']:
            if cur_df[cur_df['node1'] == node_income]['Operation'].values[0] == 'activate':
                dic_nodes_incoming[node][node_income] = "activate"
            else:
                dic_nodes_incoming[node][node_income] = "inactivate"

    return dic_nodes_incoming


def initial_model(all_nodes, initial_values):
    # Initialize the model
    model = {}
    for i in range(0, len(all_nodes)):
        model[list(all_nodes)[i]] = initial_values[i]

    return model


def node_update_rules(init_cur_upstream_values, init_cur_operation):
    '''
    Example input:
    init_cur_upstream_values = [True,False,False,True]
    init_cur_operation = ["activate","activate","activate",'inactivate']    
    '''
    # get the index of the operation where the operation is activate
    active_index = [i for i, x in enumerate(init_cur_operation) if x == "activate"]
    inactive_index = [i for i, x in enumerate(init_cur_operation) if x == "inactivate"]

   
    if len(active_index) > 0:
        cur1 = False
        for index in active_index:
            cur1 = cur1 or init_cur_upstream_values[index]
    else:
        cur1 = False

    if len(inactive_index) > 0:
        cur2 = False
        for index in inactive_index:
            cur2 = cur2 or init_cur_upstream_values[index]
    else:
        cur2 = False

    result = cur1 and (not cur2)
    return result

def xor(a, b):
    return (a and not b) or (not a and b)


def update_model_with_flip_onestep(model_t0, dic_nodes_incoming, noise):
    model_t1 = model_t0.copy()
    all_nodes = list(model_t0.keys())
    n = 0

    for cur_node in all_nodes:
        n = n + 1
        if noise[n-1] == True:
            cur_node_value =  xor(model_t0[cur_node], noise[n-1])

        else:
            if cur_node in list(dic_nodes_incoming.keys()):
                updstream_nodes =list(dic_nodes_incoming[cur_node].keys())
                init_cur_upstream_values = []
                init_cur_operation = []

                for upstream_node in updstream_nodes:
                    init_cur_upstream_values.append(model_t0[upstream_node])
                    init_cur_operation.append(dic_nodes_incoming[cur_node][upstream_node])

                cur_node_value = node_update_rules(init_cur_upstream_values,init_cur_operation )

            else:
                cur_node_value = model_t0[cur_node]

        model_t1[cur_node] = cur_node_value
        
    return(model_t1)

def update_model_with_flip_onestep_withPerterbation(model_t0, dic_nodes_incoming, noise, onlist = [], offlist = []):
    '''
    
    '''
    model_t1 = model_t0.copy()
    all_nodes = list(model_t0.keys())
    n = 0

    for cur_node in all_nodes:
        n = n + 1
        if cur_node in onlist:
            cur_node_value = True
            
        elif cur_node in offlist:
            cur_node_value = False
            

        
        else:
            if noise[n-1] == True:
                cur_node_value =  xor(model_t0[cur_node], noise[n-1])

            else:
                if cur_node in list(dic_nodes_incoming.keys()):
                    updstream_nodes =list(dic_nodes_incoming[cur_node].keys())
                    init_cur_upstream_values = []
                    init_cur_operation = []

                    for upstream_node in updstream_nodes:
                        init_cur_upstream_values.append(model_t0[upstream_node])
                        init_cur_operation.append(dic_nodes_incoming[cur_node][upstream_node])

                    cur_node_value = node_update_rules(init_cur_upstream_values,init_cur_operation )

                else:
                    cur_node_value = model_t0[cur_node]

        model_t1[cur_node] = cur_node_value
        
    return(model_t1)

def update_model_without_flip_onestep(model_t0,dic_nodes_incoming):
    model_t1 = model_t0.copy()

    for cur_node in list(model_t0.keys()):

        if cur_node in list(dic_nodes_incoming.keys()):
            updstream_nodes =list(dic_nodes_incoming[cur_node].keys())
            init_cur_upstream_values = []
            init_cur_operation = []

            for upstream_node in updstream_nodes:
                init_cur_upstream_values.append(model_t0[upstream_node])
                init_cur_operation.append(dic_nodes_incoming[cur_node][upstream_node])
            
            cur_node_value = node_update_rules(init_cur_upstream_values,init_cur_operation)
            model_t1[cur_node] = cur_node_value

        else:
            model_t1[cur_node] = model_t0[cur_node]
    
    return(model_t1)

def update_model_without_flip_onestep_withPerterbation(model_t0,dic_nodes_incoming, onlist = [], offlist = []):
    model_t1 = model_t0.copy()

    for cur_node in list(model_t0.keys()):
        if cur_node in onlist:
            cur_node_value = True
            model_t1[cur_node] = cur_node_value

        elif cur_node in offlist:
            cur_node_value = False
            model_t1[cur_node] = cur_node_value

        elif cur_node in list(dic_nodes_incoming.keys()):
            updstream_nodes =list(dic_nodes_incoming[cur_node].keys())
            init_cur_upstream_values = []
            init_cur_operation = []

            for upstream_node in updstream_nodes:
                init_cur_upstream_values.append(model_t0[upstream_node])
                init_cur_operation.append(dic_nodes_incoming[cur_node][upstream_node])
            
            cur_node_value = node_update_rules(init_cur_upstream_values,init_cur_operation)
            model_t1[cur_node] = cur_node_value

        else:
            model_t1[cur_node] = model_t0[cur_node]
    
    return(model_t1)
    
def update_models_multisteps(model_initial,dic_nodes_incoming, runs = 20, fliprop = 0.01):
    model = model_initial.copy()
    models = {}
    for node in list(model_initial.keys()):
            models[node] = [model_initial[node]]

    for run in range(0,runs): 
        all_nodes = list(model_initial.keys())
        noise = np.random.uniform(0,1,len(all_nodes))
        noise[noise < fliprop] = True  # 0.01 chance to flip the value
        noise[noise != True] = False  
    
        if True in noise:
            model_t1 = update_model_with_flip_onestep(model,dic_nodes_incoming,noise)   
        else:
            model_t1 = update_model_without_flip_onestep(model,dic_nodes_incoming)
        
        for node in list(model_t1.keys()):
            models[node].append(model_t1[node])
        
        model = model_t1.copy()

    return(models)

def update_models_multisteps_with_perterbation(model_initial,dic_nodes_incoming, runs = 20, fliprop = 0.01, onlist = [], offlist = []):
    model = model_initial.copy()
    models = {}
    for node in list(model_initial.keys()):
            models[node] = [model_initial[node]]

    for run in range(0,runs): 
        all_nodes = list(model_initial.keys())
        noise = np.random.uniform(0,1,len(all_nodes))
        noise[noise < fliprop] = True  # 0.01 chance to flip the value
        noise[noise != True] = False  
    
        if True in noise:
            model_t1 = update_model_with_flip_onestep_withPerterbation(model,dic_nodes_incoming,noise, onlist, offlist)   
        else:
            model_t1 = update_model_without_flip_onestep_withPerterbation(model,dic_nodes_incoming, onlist, offlist)
        
        for node in list(model_t1.keys()):
            models[node].append(model_t1[node])
        
        model = model_t1.copy()

    return(models)