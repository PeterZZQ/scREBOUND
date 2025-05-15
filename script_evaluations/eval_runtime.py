# In[]
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# In[]
PROJECT_DIR = "/net/csefiles/xzhanglab/zzhang834/LLM_KD/"
res_dir = PROJECT_DIR + "results/runtime/"
runtime_cum_list = []
runtime_list = []
for model_name in ["cp_contrcb1_6_512_256_encbg_level2_1", "scGPT", "UCE", "UCE_4layer", "scMulan", "geneformer", "scFoundation"]:
    runtime_cum = [0]
    cell_cum = [0]
    runtime = pd.read_csv(res_dir + f"runtime_{model_name}.csv", index_col = 0)
    # print(runtime.shape[0])
    if runtime.shape[0] > 3907:
        runtime = runtime.iloc[:3907, :]
    for it in range(runtime.shape[0]):
        new_runtime = runtime_cum[-1] + runtime.loc[it, "runtime"]
        new_ncells = cell_cum[-1] + runtime.loc[it, "ncells"]
        runtime_cum.append(new_runtime)
        cell_cum.append(new_ncells)
    
    if model_name == "UCE_4layer":
        model_name = "UCE-4LAYER"
    elif model_name == "UCE":
        model_name = "UCE-33LAYER"
    elif model_name == "geneformer":
        model_name = "GeneFormer"
    runtime["method"] = model_name
    runtime_list.append(runtime)

    runtime_cum_df = pd.DataFrame(columns = ["ncells", "runtime"])
    runtime_cum_df["ncells"] = cell_cum
    runtime_cum_df["runtime"] = runtime_cum
    runtime_cum_df["method"] = model_name
    runtime_cum_df = runtime_cum_df.iloc[100::400,:]
    runtime_cum_list.append(runtime_cum_df)

runtime_cum_list = pd.concat(runtime_cum_list, axis = 0, ignore_index = True)
runtime_list = pd.concat(runtime_list, axis = 0, ignore_index = 0)

# In[]

# runtime_cum_list.loc[runtime_cum_list["method"] == "cp_contrcb1_6_512_256_encbg_level2_1", "method"] = "scREBOUND-6LAYER"
# runtime_list.loc[runtime_list["method"] == "cp_contrcb1_6_512_256_encbg_level2_1", "method"] = "scREBOUND-6LAYER"
runtime_cum_list.loc[runtime_cum_list["method"] == "cp_contrcb1_6_512_256_encbg_level2_1", "method"] = "scREBOUND"
runtime_list.loc[runtime_list["method"] == "cp_contrcb1_6_512_256_encbg_level2_1", "method"] = "scREBOUND"
sns.set_theme(font_scale = 1.2)
fig = plt.figure(figsize = (8,10))
axs = fig.subplots(nrows = 3, ncols = 1)
sns.lineplot(runtime_cum_list, x = "ncells", y = "runtime", hue = "method", style = "method", ax = axs[0], markers=True, markersize = 6)
axs[0].set_ylabel("Runtime (sec)")
axs[0].set_yscale("log")

leg = axs[0].legend(loc='upper left', fontsize = 12, title_fontsize = 14, frameon = False, bbox_to_anchor=(1.04, 1), title = "Method")

sns.barplot(runtime_list, x = "method", y = "runtime", ax = axs[1], width = 0.4)
for container in axs[1].containers:
    axs[1].bar_label(container, fmt = "%.2f", color = "blue")
axs[1].set_xlabel(None)
axs[1].set_ylabel("Runtime (sec)")
# leg = axs[1].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Method")
axs[1].set_yscale("log")

sns.barplot(runtime_list, x = "method", y = "peak_memory", ax = axs[2], width = 0.4)
for container in axs[2].containers:
    axs[2].bar_label(container, fmt = "%.2f", color = "blue")
axs[2].set_xlabel(None)
axs[2].set_ylabel("Peak Memory (GB)")
# leg = axs[1].legend(loc='upper left', prop={'size': 10}, frameon = False, bbox_to_anchor=(1.04, 1), title = "Method")
axs[2].set_yscale("log")


plt.tight_layout()
matplotlib.rc_file_defaults()

fig.savefig(res_dir + "runtime.png", bbox_inches = "tight", dpi = 250)
# %%
