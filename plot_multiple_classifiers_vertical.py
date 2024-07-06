import matplotlib.pyplot as plt
import pandas as pd

# Data for each classifier
data_lr = {
    "Input": ["female", "black", "gay", "hispanic", "african"],
    "Classifier Predictions": [0.297301, 0.235937, 0.259741, 0.161253, 0.174501],
    "ATE Scores": [0.088221, 0.044726, 0.072054, 0.003648, 0.018100]
}
data_dt = {
    "Input": ["female", "black", "gay", "hispanic", "african"],
    "Classifier Predictions": [0.342557, 0.453448, 0.539441, 0.287052, 0.349312],
    "ATE Scores": [0.006392, 0.039215, 0.064766, 0.000403, 0.009071]
}
data_svm = {
    "Input": ["female", "black", "gay", "hispanic", "african"],
    "Classifier Predictions": [0.337006, 0.267362, 0.265001, 0.275313, 0.248376],
    "ATE Scores": [0.078320, 0.026489, 0.026705, 0.026089, 0.021634]
}

# Creating DataFrames
df_lr = pd.DataFrame(data_lr)
df_dt = pd.DataFrame(data_dt)
df_svm = pd.DataFrame(data_svm)

# Pastel color schemes for each plot
# colors_lr = ['#add8e6', '#b0e0e6']
colors_lr = ['#10a37f', '#147960']
# colors_dt = ['#98fb98', '#90ee90']
colors_dt = ['#99ccff', '#3399ff']
colors_svm = ['#ffb6c1', '#ff69b4'] # Light pink shades for SVM

# Font sizes
title_fontsize = 14
axes_label_fontsize = 12
tick_fontsize = 11
legend_fontsize = 11

# Creating the bar plots with the updated pastel color scheme and font sizes
fig, axes = plt.subplots(3, 1, figsize=(4, 8), dpi=199)  # Adjusting DPI for better resolution

# Logistic Regression
axes[0].bar(df_lr["Input"], df_lr["Classifier Predictions"], width=0.4, color=colors_lr[0], align='center', label='LR Predictions')
axes[0].bar(df_lr["Input"], df_lr["ATE Scores"], width=0.4, color=colors_lr[1], align='edge', label='LR ATE Scores')
axes[0].set_title('Logistic Regression', fontsize=title_fontsize)
axes[0].set_ylabel('Scores', fontsize=axes_label_fontsize)
axes[0].tick_params(labelsize=tick_fontsize)
axes[0].legend(fontsize=legend_fontsize)

# Decision Tree
axes[1].bar(df_dt["Input"], df_dt["Classifier Predictions"], width=0.4, color=colors_dt[0], align='center', label='Decision Tree Predictions')
axes[1].bar(df_dt["Input"], df_dt["ATE Scores"], width=0.4, color=colors_dt[1], align='edge', label='Decision Tree ATE Scores')
axes[1].set_title('Decision Tree', fontsize=title_fontsize)
axes[1].tick_params(labelsize=tick_fontsize)
axes[1].legend(fontsize=legend_fontsize)

# SVM
axes[2].bar(df_svm["Input"], df_svm["Classifier Predictions"], width=0.4, color=colors_svm[0], align='center', label='SVM Predictions')
axes[2].bar(df_svm["Input"], df_svm["ATE Scores"], width=0.4, color=colors_svm[1], align='edge', label='SVM ATE Scores')
axes[2].set_title('Support Vector Machine (SVM)', fontsize=title_fontsize)
axes[2].tick_params(labelsize=tick_fontsize)
axes[2].legend(fontsize=legend_fontsize)

plt.tight_layout()
plt.savefig("plots/multiple_classifiers_vertical.png", dpi=199)
plt.savefig("plots/multiple_classifiers_vertical.svg", dpi=199)
plt.savefig("plots/multiple_classifiers_vertical.pdf", dpi=199)
plt.show()
