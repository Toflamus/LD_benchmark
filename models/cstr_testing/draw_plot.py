import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cstr import build_model

def solve_scenario(num_reactors, recycle_loc, max_nt=8):
    # Create model with fixed superstructure size
    m = build_model(NT=max_nt)
    
    # Apply transformations
    pyo.TransformationFactory("core.logical_to_linear").apply_to(m)
    pyo.TransformationFactory("gdp.bigm").apply_to(m)
    
    # Fix Chain Length (YF)
    # If num_reactors = k, then YF[k] = True, others False
    for n in m.N:
        if n == num_reactors:
            m.YF[n].fix(True)
        else:
            m.YF[n].fix(False)
            
    # Fix Recycle Location (YR)
    # If recycle_loc = r, then YR_is_recycle[r] is active
    # Note: In the transformed model, the indicator var handles the logic.
    # We need to find the binary variable associated with the disjunct.
    # The code uses m.YR_is_recycle[n].indicator_var
    for n in m.N:
        if n == recycle_loc:
            m.YR_is_recycle[n].indicator_var.fix(True)
            m.YR_is_not_recycle[n].indicator_var.fix(False)
        else:
            m.YR_is_recycle[n].indicator_var.fix(False)
            m.YR_is_not_recycle[n].indicator_var.fix(True)
            
    # Solve
    solver = pyo.SolverFactory('gams')
    try:
        # Suppress output to keep log clean
        res = solver.solve(m, tee=False)
        
        if (res.solver.status == pyo.SolverStatus.ok) and \
           (res.solver.termination_condition == pyo.TerminationCondition.optimal):
            return pyo.value(m.obj)
        else:
            return np.nan
    except Exception as e:
        return np.nan

# Grid search 1-8 * 1-8
results = []
max_nt = 8

for k in range(1, max_nt+1): # Number of reactors
    for r in range(1, max_nt+1): # Recycle location
        if r <= k: # Recycle must be within the active chain
            val = solve_scenario(k, r, max_nt)
            results.append({'Num_Reactors': k, 'Recycle_Location': r, 'Total_Volume': val})
        else:
            results.append({'Num_Reactors': k, 'Recycle_Location': r, 'Total_Volume': np.nan})

df = pd.DataFrame(results)

# Create Pivot for Heatmap
pivot_table = df.pivot(index='Num_Reactors', columns='Recycle_Location', values='Total_Volume')

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".4f")
plt.title('Total Reactor Volume for CSTR Configurations')
plt.ylabel('Number of Reactors (Chain Length)')
plt.xlabel('Recycle Location (Reactor Index)')
plt.gca().invert_yaxis() # Usually index 1 is top, but let's keep 1 at bottom for intuitive "size"
plt.savefig('cstr_optimization_landscape.png')

print("Grid search complete.")
print(df.head(25))
df.to_csv('cstr_grid_results.csv', index=False)