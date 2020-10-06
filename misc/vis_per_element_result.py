#  Copyright (c) 2020
#  Licensed under The MIT License
#  Written by Zhiheng Li
#  Email: zhiheng.li@rochester.edu


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


element_to_frequency = {'C': 22150, 'N': 4418, 'O': 3930, 'S': 407, 'Cl': 296, 'F': 352, 'Br': 172, 'B': 17, 'P': 20, 'I': 53, 'Si': 36, 'Se': 5, 'Sn': 2, 'Ru': 2, 'K': 2, 'Fe': 1}
# {'C': 39563488, 'N': 6113034, 'O': 6278194, 'S': 711604, 'Cl': 406332, 'F': 700548, 'Br': 84706, 'P': 40590,
#                  'Se': 2290, 'B': 3374, 'I': 12108, 'Si': 3269, 'Te': 103, 'As': 205, 'Zn': 4, 'Al': 12}
# elements = list(element_to_frequency.keys())
elements = ['I', 'S', 'F', 'N', 'O', 'P', 'Br', 'Si', 'Se', 'B', 'Cl', 'C']

elements.sort(key=lambda x: element_to_frequency[x], reverse=True)

pkl_fpath = '/scratch/zli82/cg_exp/per_element/iris_pos_pair_weight_0.1.pkl'
with open(pkl_fpath, 'rb') as f:
    p1 = pickle.load(f)

# sns.set_theme(style="darkgrid")
#
# penguins = sns.load_dataset("penguins")
#
# # Draw a nested barplot by species and sex
# g = sns.catplot(
#     data=penguins, kind="bar",
#     x="species", y="body_mass_g", hue="sex",
#     ci="sd", palette="dark", alpha=.6, height=6
# )
# g.despine(left=True)
# g.set_axis_labels("", "Body mass (g)")
# g.legend.set_title("")



pkl_fpath = '/scratch/zli82/cg_exp/per_element/private_ChEMBL_weighted_ft_0_05_sampled_ratio_1.0.pkl'
with open(pkl_fpath, 'rb') as f:
    p2 = pickle.load(f)


ami_lst = [p2[e]['mean']['ami'] - p1[e]['mean']['ami'] for e in elements]
cut_prec_lst = [p2[e]['mean']['cut_prec'] - p1[e]['mean']['cut_prec'] for e in elements]
cut_recall_lst = [p2[e]['mean']['cut_recall'] - p1[e]['mean']['cut_recall'] for e in elements]
cut_fscore_lst = [p2[e]['mean']['cut_fscore'] - p1[e]['mean']['cut_fscore'] for e in elements]

x = np.arange(len(elements)) * 2  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5 * width, ami_lst, width, label='AMI')
rects2 = ax.bar(x - width/2, cut_prec_lst, width, label='Cut Precision')
rects3 = ax.bar(x + width/2, cut_prec_lst, width, label='Cut Recall')
rects4 = ax.bar(x + 1.5 * width, cut_prec_lst, width, label='Cut F1-Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('w/ Pre-training Result - w/o Pretraining Result')
ax.set_title('Per Element Result difference on HAM dataset')
ax.set_xticks(x)
ax.set_xticklabels(elements)
ax.legend(loc='lower right')


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)

# fig.tight_layout()

plt.show()
pass