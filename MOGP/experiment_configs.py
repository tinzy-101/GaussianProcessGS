
EXPERIMENT_CONFIGS = {
    # Default (Independent) - no specific rank
    "independent": {
        "model_class": "IndependentMultiTaskGPModel",
        "params": {"nu": 0.5},
        "description": "Independent multitask GP (non-separable)"
    },
    
    # Matern 0.5 with ranks 1-6
    "matern05_rank1": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 1, "nu": 0.5},
        "description": "LCM with Matern 0.5, rank 1"
    },
    "matern05_rank2": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 2, "nu": 0.5},
        "description": "LCM with Matern 0.5, rank 2"
    },
    "matern05_rank3": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 3, "nu": 0.5},
        "description": "LCM with Matern 0.5, rank 3"
    },
    "matern05_rank4": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 4, "nu": 0.5},
        "description": "LCM with Matern 0.5, rank 4"
    },
    "matern05_rank5": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 5, "nu": 0.5},
        "description": "LCM with Matern 0.5, rank 5"
    },
    "matern05_rank6": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 6, "nu": 0.5},
        "description": "LCM with Matern 0.5, rank 6"
    },
    
    # Matern 1.5 with top 3 ranks from Matern 0.5
    "matern15_rank2": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 2, "nu": 1.5},
        "description": "LCM with Matern 1.5, rank 2"
    },
    "matern15_rank3": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 3, "nu": 1.5},
        "description": "LCM with Matern 1.5, rank 3"
    },
    "matern15_rank4": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 4, "nu": 1.5},
        "description": "LCM with Matern 1.5, rank 4"
    },
    
    # Matern 2.5 with top 3 ranks from Matern 0.5
    "matern25_rank2": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 2, "nu": 2.5},
        "description": "LCM with Matern 2.5, rank 2"
    },
    "matern25_rank3": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 3, "nu": 2.5},
        "description": "LCM with Matern 2.5, rank 3"
    },
    "matern25_rank4": {
        "model_class": "LCMMultiTaskGPModel",
        "params": {"rank": 4, "nu": 2.5},
        "description": "LCM with Matern 2.5, rank 4"
    },
    
    # Mixed Matern 0.5/2.5 with rank 1
    "mixed_matern_rank1": {
        "model_class": "LCMMixedMaternModel",
        "params": {"rank": 1},
        "description": "LCM with mixed Matern 0.5/2.5, rank 1"
    },
    
    # Mixed Matern 0.5/2.5 with rank 3
    "mixed_matern_rank3": {
        "model_class": "LCMMixedMaternModel",
        "params": {"rank": 3},
        "description": "LCM with mixed Matern 0.5/2.5, rank 3"
    },
    
    # Matern 2.5 + Wendland with rank 3
    "matern25_wendland_rank3": {
        "model_class": "LCMMaternWendlandModel",
        "params": {"rank": 3},
        "description": "LCM with Matern 2.5 + Wendland, rank 3"
    }
}