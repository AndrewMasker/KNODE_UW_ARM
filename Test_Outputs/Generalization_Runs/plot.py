import matplotlib.pyplot as plt
import json

#new_sums = []
old_sums = []
#fourier_sums = []
fsindy_sums = []
gp_sums = []

inds = [100 , 1000 , 10000 , 50000]

    
    
for ind in inds:
    path_to_val_json = "Test_Outputs/Generalization_Runs/Old_" + str(ind) + "/validation_statistics/summary.json"
    with open(path_to_val_json) as f:
        d = json.load(f)
    summary = d
    old_sums.append(summary)
    
    
for ind in inds:
    path_to_val_json = "Test_Outputs/Generalization_Runs/fsindy_" + str(ind) + "/validation_statistics/summary.json"
    with open(path_to_val_json) as f:
        d = json.load(f)
    summary = d
    fsindy_sums.append(summary)
    
    
for ind in inds:
    path_to_val_json = "Test_Outputs/Generalization_Runs/gp_" + str(ind) + "/validation_statistics/summary.json"
    with open(path_to_val_json) as f:
        d = json.load(f)
    summary = d
    gp_sums.append(summary)
    
    
old_qpos_rmse = [summary["learned_qpos_rmse"] for summary in old_sums]
nom_qpos_rmse = [summary["nominal_qpos_rmse"] for summary in old_sums]
fsindy_qpos_rmse = [summary["learned_qpos_rmse"] for summary in fsindy_sums]
gp_qpos_rmse = [summary["learned_qpos_rmse"] for summary in gp_sums]

fig , (ax1,ax2 ) = plt.subplots(1 ,2)

fig.suptitle("Statistics vs. Training Samples")
ax1.plot(inds , old_qpos_rmse , label = "KNODE")
ax1.plot(inds , nom_qpos_rmse , label = "Nominal")
ax1.plot(inds[2:] , fsindy_qpos_rmse[2:] , label = "SINDy")
ax1.plot(inds , gp_qpos_rmse , label = "GP")
ax1.set_xscale('log')
ax1.set_ylim([0, None])
ax1.set_ylabel("RMSE")
ax1.set_xlabel("Training Samples")
ax1.legend()
ax1.set_box_aspect(1)

#ax2.loglog(inds , old_qpos_rmse , label = "KNODE")
#ax2.loglog(inds , nom_qpos_rmse , label = "Nominal")
#ax2.loglog(inds , fsindy_qpos_rmse , label = "SINDy")
#ax2.loglog(inds , gp_qpos_rmse , label = "GP")
#ax1.set_ylabel("q RMSE")
#ax1.set_xlabel("Training Samples")
#ax2.legend()
#ax2.set_box_aspect(1)





#old_qpos_rmse = [summary["learned_qvel_rmse"] for summary in old_sums]
#nom_qpos_rmse = [summary["nominal_qvel_rmse"] for summary in old_sums]
#fsindy_qpos_rmse = [summary["learned_qvel_rmse"] for summary in fsindy_sums]
#gp_qpos_rmse = [summary["learned_qvel_rmse"] for summary in gp_sums]
#
#fig , (ax1) = plt.subplots(1 , 1)
#
#fig.suptitle("q_dot RMSE vs. Training Samples")
#ax1.plot(inds , old_qpos_rmse , label = "KNODE")
#ax1.plot(inds , nom_qpos_rmse , label = "Nominal")
#ax1.plot(inds[2:] , fsindy_qpos_rmse[2:] , label = "SINDy")
#ax1.plot(inds , gp_qpos_rmse , label = "GP")
#ax1.set_xscale('log')
#ax1.set_ylabel("q_dot RMSE")
#ax1.set_xlabel("Training Samples")
#ax1.set_ylim([0, None])
#ax1.legend()
#ax1.set_box_aspect(1)

#ax2.loglog(inds , old_qpos_rmse , label = "KNODE")
#ax2.loglog(inds , nom_qpos_rmse , label = "Nominal")
#ax2.loglog(inds[2:] , fsindy_qpos_rmse[2:] , label = "SINDy")
#ax2.loglog(inds , gp_qpos_rmse , label = "GP")
#ax1.set_ylabel("q_dot RMSE")
#ax1.set_xlabel("Training Samples")
#ax2.legend()
#ax2.set_box_aspect(1)




old_qpos_rmse = [summary["qpos_improvement_pct"] for summary in old_sums]
fsindy_qpos_rmse = [summary["qpos_improvement_pct"] for summary in fsindy_sums]
gp_qpos_rmse = [summary["qpos_improvement_pct"] for summary in gp_sums]

#fig , (ax1 ) = plt.subplots(1 , 1)

#fig.suptitle("q RMSE Percent Improvement vs. Training Samples")
ax2.plot(inds , old_qpos_rmse , label = "KNODE")
ax2.plot(inds[2:] , fsindy_qpos_rmse[2:] , label = "SINDy")
ax2.plot(inds , gp_qpos_rmse , label = "GP")
ax2.set_xscale('log')
ax2.set_ylabel("RMSE % Improvement")
ax2.set_xlabel("Training Samples")
ax2.legend()
ax2.set_box_aspect(1)

#ax2.loglog(inds , old_qpos_rmse , label = "KNODE")
#ax2.loglog(inds[2:] , fsindy_qpos_rmse[2:] , label = "SINDy")
#ax2.loglog(inds , gp_qpos_rmse , label = "GP")
#ax1.set_ylabel("q RMSE Percent Improvement")
#ax1.set_xlabel("Training Samples")
#ax2.legend()
#ax2.set_box_aspect(1)








plt.show()
