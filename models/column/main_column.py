from __future__ import division
from pickle import TRUE
import sys
sys.path.append('C:/Users/dlinanro/Desktop/mld-bd/') 
from functions.d_bd_functions import run_function_dbd
from functions.dsda_functions import neighborhood_k_eq_inf
from Design.gdp_dist_model_design import build_column
import logging
from functions.dsda_functions import get_external_information,external_ref
import pyomo.environ as pe




def problem_logic_column(m):
    logic_expr = []
    for n in m.intTrays:
        logic_expr.append([pe.land(~m.YR[n] for n in range(
            m.reboil_tray+1, m.feed_tray)), m.YR_is_down])
        logic_expr.append([pe.land(~m.YB[n]
                                   for n in range(m.feed_tray+1, m.max_trays)), m.YB_is_up])
    for n in m.conditional_trays:
        logic_expr.append([pe.land(pe.lor(m.YR[j] for j in range(n, m.max_trays)), pe.lor(
            pe.land(~m.YB[j] for j in range(n, m.max_trays)), m.YB[n])), m.tray[n].indicator_var])
        logic_expr.append([~pe.land(pe.lor(m.YR[j] for j in range(n, m.max_trays)), pe.lor(
            pe.land(~m.YB[j] for j in range(n, m.max_trays)), m.YB[n])), m.no_tray[n].indicator_var])
    return logic_expr




if __name__ == "__main__":
    #Do not show warnings
    logging.getLogger('pyomo').setLevel(logging.ERROR)
    
    ###REFORMUALTION EXTERNAL VARIABLES
    kwargs={'min_trays':8,'max_trays':17,'xD':0.95,'xB':0.95}
    model_fun=build_column
    logic_fun=problem_logic_column
    model =build_column(**kwargs)
    ext_ref = {model.YB: model.intTrays, model.YR: model.intTrays} #reformulation sets and variables
    reformulation_dict, number_of_external_variables, lower_bounds, upper_bounds = get_external_information(model, ext_ref, tee=True) 


    initialization=[13,4] 
    infinity_val=1e+9
    nlp_solver='knitro'
    neigh=neighborhood_k_eq_inf(2)
    maxiter=100
    sub_options={}


    [important_info,important_info_preprocessing,D,x_actual,m]=run_function_dbd(initialization,infinity_val,nlp_solver,neigh,maxiter,ext_ref,logic_fun,model_fun,kwargs,use_random=False,sub_solver_opt=sub_options, tee=True,one_cut_per_iteration=True,updated_stop_condition=False)
    print('Objective value: ',str(pe.value(m.obj)))
    print('Objective value: ',str(important_info['m3_s3'][0])+'; time= ',str(important_info['m3_s3'][1]))
