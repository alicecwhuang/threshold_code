import AgentsModels
import pandas as pd

n = 3000
d = pd.DataFrame()
for i in range(5, 16):
	m = AgentsModels.epistemicModel(N_FullPure=n, N_SmallPure=n, N_FullSens=n, N_cos=n, N_SmallSens=n, N_hyp=i)
	m.step()
	d = pd.concat([d, m.datacollector.get_agent_vars_dataframe().dropna()])
d.to_csv("data.csv", index=False)
