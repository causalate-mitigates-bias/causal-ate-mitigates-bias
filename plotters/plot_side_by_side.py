import os
import sys

# Add the project root to the system path
project_root = os.getcwd()
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import pandas as pd

# Data for Zampieri et al., 2019 dataset
data_zampieri = {
    "Input": ["female", "black", "gay", "hispanic", "african"],
    "Classifier Predictions": [0.298005, 0.235770, 0.259882, 0.161886, 0.174550],
    "ATE Scores": [0.088414, 0.044404, 0.071427, 0.003732, 0.017795]
}
df_zampieri = pd.DataFrame(data_zampieri)

# Data for Gao et al., 2019 dataset
data_gao = {
    "Input": ["female", "black", "gay", "hispanic", "african"],
    "Classifier Predictions": [0.269563, 0.300420, 0.470384, 0.165930, 0.201015],
    "ATE Scores": [0.167263, 0.107566, 0.167032, 0.011108, 0.099372]
}
df_gao = pd.DataFrame(data_gao)

# Reordering the DataFrame based on the alphabetical order of 'Input'
df_zampieri = df_zampieri.sort_values(by='Input')
df_gao = df_gao.sort_values(by='Input')

# Plotting with updated y-axis label and ordered x-axis
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Font sizes
title_fontsize = 14
axes_label_fontsize = 12
tick_fontsize = 11
legend_fontsize = 11

# Define the color scheme
light_green_color = '#10a37f'      # '#008000'
darker_green_color = '#147960'  # Or a lighter green code like '#90ee90'


# Zampieri et al., 2019
axes[0].bar(df_zampieri["Input"], df_zampieri["Classifier Predictions"], width=0.4, color=light_green_color, align='center', label='Classifier Predictions')
axes[0].bar(df_zampieri["Input"], df_zampieri["ATE Scores"], width=0.4, color=darker_green_color, align='edge', label='ATE Scores')
axes[0].set_title('Zampieri et al., 2019 Dataset', fontsize=title_fontsize)
axes[0].set_ylabel('Toxicity Scores', fontsize=axes_label_fontsize)
axes[0].legend(fontsize=legend_fontsize)

# Gao et al., 2019
axes[1].bar(df_gao["Input"], df_gao["Classifier Predictions"], width=0.4, color=light_green_color, align='center', label='Classifier Predictions')
axes[1].bar(df_gao["Input"], df_gao["ATE Scores"], width=0.4, color=darker_green_color, align='edge', label='ATE Scores')
axes[1].set_title('Gao et al., 2017 Dataset', fontsize=title_fontsize)
axes[1].set_ylabel('Toxicity Scores', fontsize=axes_label_fontsize)
axes[1].legend(fontsize=legend_fontsize)

plt.tight_layout()
plt.savefig("plots/difference_in_bias_side_by_side.png", dpi=199)
plt.savefig("plots/difference_in_bias_side_by_side.svg", dpi=199)
plt.show()
