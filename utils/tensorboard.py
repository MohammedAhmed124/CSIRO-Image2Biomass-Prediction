import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from io import BytesIO
from .config import CFG
import subprocess
from typing import Optional
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

def luanch_tensorboard(
        output_dir="kf_runs",
        continue_=False,
        purge_step: Optional[int] = None
        ):
    """
    Launch a SummaryWriter pointed at output_dir/tensorboard_global.
    If continue_==False the log_dir is removed first.
    purge_step: optional int passed to SummaryWriter to avoid step overlap.
    """
    subprocess.run(["pkill", "-f", "tensorboard"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    log_dir = os.path.join(output_dir, "tensorboard_global")
    if os.path.exists(log_dir) and (not continue_):
        shutil.rmtree(log_dir)

    # create parent if missing
    os.makedirs(log_dir, exist_ok=True)

    # pass purge_step so new writer won't overlap old steps
    writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step)

    command = [
        "tensorboard",
        "--logdir", log_dir,
        "--port", str(6006),
        "--reload_interval", "1"
    ]

    tensorboard_process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Launching TensorBoard at http://localhost:{6006}/ ...")
    return writer




def df_to_image(
        df,
        title=None,
        font_size=9,
        header_font_size=10
        ):
    fig, ax = plt.subplots(figsize=(min(12, len(df.columns)*1.2), 0.5*len(df)+1))
    ax.axis('off')
    df = df.copy()

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    
    # Style headers
    for j in range(len(df.columns)):
        cell = table[(0, j)]
        cell.set_text_props(weight='bold', fontsize=header_font_size)
        cell.set_facecolor('#dddddd')
        cell.set_edgecolor('black')
    
    if title:
        plt.title(title, fontsize=header_font_size+2, pad=15)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    arr = np.array(img)
    tensor = torch.tensor(arr).permute(2, 0, 1).float() / 255
    return tensor

def split_df_into_groups(
        df,
        group_size=5
        ):
    n_cols = df.shape[1]
    groups = []
    for i in range(0, n_cols, group_size):
        sub_df = df.iloc[:, i:i+group_size]
        target = [target for target in CFG.ALL_TARGETS + ["total"] if target in sub_df.columns[0]][0]
        sub_df.columns = [column.rstrip("_" + target) for column in sub_df.columns]
        groups.append((sub_df , target))
    return groups 