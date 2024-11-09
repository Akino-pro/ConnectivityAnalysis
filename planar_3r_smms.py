# https://github.com/landonclark97/numbotics.git


import numbotics.robot as rob
import numbotics.spatial as spt
import numbotics.topology as top
import numpy as np

arm = rob.Robot('3R.rob')

samples = 100

for x in np.linspace(0.0, arm.length, samples):
    x_d = spt.trans_mat(pos=np.array([[x, 0.0, 0.0]]).T)

    # smms : (B, S, n)
    s, smms = arm.all_smms(x_d)
    print(smms[0,0,0])
    smms = top.tr_inv(smms)
    print(smms[0,0,0])


    if s:
        print(f'solved for {smms.shape[0]} self-motion manifolds at location: {x_d}')
