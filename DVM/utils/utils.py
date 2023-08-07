import matplotlib.pyplot as plt
import numpy as np
import os
def compare(ori, pre, prediction_point, title_list):
    for index_image in range(8):
        fig, axs = plt.subplots(4, 2, figsize=(15, 15))
        for i, ax in enumerate(axs.flatten()):
            index = index_image * 8 + i
            title = title_list[index]
            x = np.arange(1, 65)
            ori_data = ori[:, index]
            pre_data = pre[:, index]
            ax.plot(x, ori_data, label='original data')
            ax.plot(x, pre_data, label='predict data')
            ax.axvline(x=prediction_point, color='r', linestyle='--')
            ax.set_title(title)
            ax.legend()
            ax.margins(0, 4)
        plt.tight_layout()
        father_path = "caches/prediction"
        if not os.path.exists(father_path):
            os.makedirs(father_path)
        plt.savefig(os.path.join(father_path,f"{index_image}.png"))
        plt.close()