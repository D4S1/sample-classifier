import wandb
import classifier
import evaluate
import psutil
import time
from threading import Thread


def evaluate_model(config=None):
    """
    Evaluate the model with the given parameters and log memory usage.
    
    :param config: A dictionary containing the parameters for the run.
    """
    with wandb.init(config=config):
        config = wandb.config

        # Parameters from the sweep
        k = config.k
        ci = config.ci
        threshold = config.threshold
        M = config.M
        human_sketch_size = config.human_sketch_size
        ref_sketch_size = config.ref_sketch_size
        sketch_size = config.sketch_size

        # Start tracking memory usage in a separate thread
        mem_usage = []
        keep_tracking = True

        def track_memory():
            while keep_tracking:
                mem_usage.append(psutil.Process().memory_info().rss / (1024 ** 2))  # convert to MiB
                time.sleep(1)  # log every second
 
        memory_thread = Thread(target=track_memory)
        memory_thread.start()

        try:
            # Evaluate the model
            auc = average_auc(k=k, ci=ci, M=M, threshold=threshold, human_sketch_size=human_sketch_size, ref_sketch_size=ref_sketch_size, sketch_size=sketch_size)
        
        finally:
            # Stop memory tracking
            keep_tracking = False
            memory_thread.join()

        # Memory logging
        max_memory = max(mem_usage) if mem_usage else 0

        # Penalized AUC: Penalize configurations that exceed 1 GB (1024 MiB)
        penalty = 0 if max_memory <= 1024 else (max_memory - 1024) / 1024
        penalized_auc = max(auc - penalty, 0.5)

        # Log metrics
        wandb.log({
            "auc": auc,
            "max_memory_usage_mib": max_memory,
            "penalized_auc": penalized_auc
        })


def average_auc(k, ci, M, threshold, human_sketch_size, ref_sketch_size, sketch_size):
    """
    Run the classifier and calculate AUC.
    """
    classifier.main('full_data/training_data.tsv', 'full_data/testing_data.tsv', 'full_data/outs/run2.tsv', k=k, ci=ci, M=M, threshold=threshold, human_sketch_size=human_sketch_size, ref_sketch_size=ref_sketch_size, sketch_size=sketch_size)
    return evaluate.calculate_auc_roc('full_data/outs/run2.tsv', 'full_data/testing_ground_truth.tsv')


# Sweep configuration for weighted metric optimization
sweep_config = {
    "method": "bayes",
    "metric": {"name": "penalized_auc", "goal": "maximize"},
    "parameters": {
        "k": {"values": list(range(5, 32, 2))},
        "ci": {"values": list(range(1, 7))},
        "threshold": {"min": 0.1, "max": 0.4},
        "M": {"values": [2, 3, 4, 5]},
        "human_sketch_size": {"values": [500, 10**3, 5*10**3, 10**4, 5*10**4, 10**5]}, 
        "ref_sketch_size": {"values": [500, 10**3, 5*10**3, 10**4, 5*10**4, 10**5]},
        "sketch_size": {"values": [500, 10**3, 5*10**3, 10**4, 5*10**4]}
    }
}

wandb.login()

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='MetagenomicsClassifier')

# Run sweep
wandb.agent(sweep_id, function=evaluate_model)
