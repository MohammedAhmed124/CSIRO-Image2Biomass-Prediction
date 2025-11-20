import os

class CFG:
    """Configuration class for all parameters."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    IMAGE_DIR = os.path.join(DATA_DIR, "train")


    MODEL_TARGETS = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g']
    ALL_TARGETS = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g']
    
    # Weights for the final R2 score
    R2_WEIGHTS = {
        "Dry_Total_g": 0.5,
        "GDM_g": 0.2,
        "Dry_Green_g": 0.1,
        "Dry_Clover_g": 0.1,
        "Dry_Dead_g": 0.1,
    }
    EPS = 1e-6

    SEED = 42