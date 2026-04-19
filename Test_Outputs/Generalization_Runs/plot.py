import matplotlib.pyplot as plt
import json

new_sums = []
old_sums = []
fourier_sums = []
fsindy_sums = []

inds = [100 , 1000 , 10000 , 50000]


for ind in inds:
    path_to_val_json = "Test_Outputs/Generalization_Runs/New_" + str(ind) + "/validation_statistics/summary.json"
    with open(path_to_val_json) as f:
        d = json.load(f)
    summary = d
    new_sums.append(summary)
    
    
for ind in inds:
    path_to_val_json = "Test_Outputs/Generalization_Runs/Old_" + str(ind) + "/validation_statistics/summary.json"
    with open(path_to_val_json) as f:
        d = json.load(f)
    summary = d
    old_sums.append(summary)
    
for ind in inds:
    path_to_val_json = "Test_Outputs/Generalization_Runs/Fourier_" + str(ind) + "/validation_statistics/summary.json"
    with open(path_to_val_json) as f:
        d = json.load(f)
    summary = d
    fourier_sums.append(summary)
    
    
for ind in inds:
    path_to_val_json = "Test_Outputs/Generalization_Runs/fsindy_" + str(ind) + "/validation_statistics/summary.json"
    with open(path_to_val_json) as f:
        d = json.load(f)
    summary = d
    fsindy_sums.append(summary)
    
    
    
new_qpos_rmse = [summary["learned_qpos_rmse"] for summary in new_sums]
old_qpos_rmse = [summary["learned_qpos_rmse"] for summary in old_sums]
nom_qpos_rmse = [summary["nominal_qpos_rmse"] for summary in new_sums]
fourier_qpos_rmse = [summary["learned_qpos_rmse"] for summary in fourier_sums]
fsindy_qpos_rmse = [summary["learned_qpos_rmse"] for summary in fsindy_sums]


fig , (ax1 , ax2) = plt.subplots(1 , 2)

ax1.plot(inds , new_qpos_rmse , label = "new")
ax1.plot(inds , old_qpos_rmse , label = "old")
ax1.plot(inds , nom_qpos_rmse , label = "nominal")
ax1.plot(inds , fourier_qpos_rmse , label = "fourier")
ax1.plot(inds , fsindy_qpos_rmse , label = "fsindy")
#ax1 = plt.gca()
#ax.set_xlim([xmin, xmax])
ax1.set_ylim([0, None])
ax1.legend()

ax2.loglog(inds , new_qpos_rmse , label = "new")
ax2.loglog(inds , old_qpos_rmse , label = "old")
ax2.loglog(inds , nom_qpos_rmse , label = "nominal")
ax2.loglog(inds , fourier_qpos_rmse , label = "fourier")
ax2.loglog(inds , fsindy_qpos_rmse , label = "fsindy")
#ax1 = plt.gca()
#ax.set_xlim([xmin, xmax])
#ax2.set_ylim([0, None])
ax2.legend()




new_qpos_rmse = [summary["learned_qvel_rmse"] for summary in new_sums]
old_qpos_rmse = [summary["learned_qvel_rmse"] for summary in old_sums]
nom_qpos_rmse = [summary["nominal_qvel_rmse"] for summary in new_sums]
fourier_qpos_rmse = [summary["learned_qvel_rmse"] for summary in fourier_sums]
fsindy_qpos_rmse = [summary["learned_qvel_rmse"] for summary in fsindy_sums]


fig , (ax1 , ax2) = plt.subplots(1 , 2)

ax1.plot(inds , new_qpos_rmse , label = "new")
ax1.plot(inds , old_qpos_rmse , label = "old")
ax1.plot(inds , nom_qpos_rmse , label = "nominal")
ax1.plot(inds , fourier_qpos_rmse , label = "fourier")
ax1.plot(inds , fsindy_qpos_rmse , label = "fsindy")
#ax1 = plt.gca()
#ax.set_xlim([xmin, xmax])
ax1.set_ylim([0, None])
ax1.legend()

ax2.loglog(inds , new_qpos_rmse , label = "new")
ax2.loglog(inds , old_qpos_rmse , label = "old")
ax2.loglog(inds , nom_qpos_rmse , label = "nominal")
ax2.loglog(inds , fourier_qpos_rmse , label = "fourier")
ax2.loglog(inds , fsindy_qpos_rmse , label = "fsindy")
#ax1 = plt.gca()
#ax.set_xlim([xmin, xmax])
#ax2.set_ylim([0, None])
ax2.legend()



plt.show()
