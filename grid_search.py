import wandb
import classifier
import evaluate


def evaluate_model(config=None):
    """
    Evaluate the model with the given parameters.
    
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

        auc = average_auc(k=k, ci=ci, M=M, threshold=threshold, human_sketch_size=human_sketch_size, ref_sketch_size=ref_sketch_size, sketch_size=sketch_size)
        
        # Log the results
        wandb.log({"auc": auc})

def average_auc(k, ci, M, threshold, human_sketch_size, ref_sketch_size, sketch_size):
    classifier.main('full_data/training_data.tsv', 'full_data/testing_data.tsv', 'full_data/outs/run2.tsv', k=k, ci=ci, M=M, threshold=threshold, human_sketch_size=human_sketch_size, ref_sketch_size=ref_sketch_size, sketch_size=sketch_size)
    return evaluate.calculate_auc_roc('full_data/outs/run2.tsv', 'full_data/testing_ground_truth.tsv')

    
sweep_config = {
    "method": "bayes",
    "metric": {"name": "auc", "goal": "maximize"},
    "parameters": {
        "k": {"values": [15, 19, 23, 27, 31]},
        "ci": {"values": [1, 2, 3, 4]},
        "threshold": {"min": 0.2, "max": 0.3},
        "M": {"values": [1, 2, 3, 4, 5]},
        "human_sketch_size": {"values": [10**3, 5*10**3, 10**4, 5*10**4, 10**5]}, 
        "ref_sketch_size": {"values": [10**3, 5*10**3, 10**4, 5*10**4, 10**5]},
        "sketch_size": {"values": [10**3, 5*10**3, 10**4, 5*10**4, 10**5]}
    }
}

wandb.login()

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='Sample-Classifier')

# Run sweep
wandb.agent(sweep_id, function=evaluate_model)
