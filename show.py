import json
import matplotlib.pyplot as plt

with open("work_dirs/oriented_reppoints_r50_fpn_1x_dota_le135/20221203_115733.log.json", 'r') as f:
    i = 0
    x = []
    loss = []
    # acc = []
    for l in f.readlines():
        d = json.loads(l)
        model = d.get('mode')
        if model is not None and model == 'train':
            # print(type())
            x.append((d["epoch"]-1)*6400+d["iter"])
            loss.append(d["loss"])
            # acc.append(d['decode.acc_seg'])
    # plt.plot(x, loss,lw = 1,c='r', mfc='w')
    plt.plot(x, loss, lw=1)
    
    plt.savefig("a.png")
