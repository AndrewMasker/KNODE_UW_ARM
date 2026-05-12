import matplotlib.pyplot as plt
import json
import numpy as np



file_path = "Models/test_uvms/loss_list.json"
#file_path = "Models/Generalization_Runs/Old_50000/loss_list.json"



with open(file_path) as f:
    d = json.load(f)

loss_list = d

fig , (ax1 , ax2) = plt.subplots(1 , 2)

loss_inds = np.arange(1 , len(loss_list) + 1)
ax1.plot(loss_inds , loss_list)
ax2.loglog(loss_inds , loss_list)

plt.show()
