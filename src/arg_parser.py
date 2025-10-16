import argparse


def get_arguments():
    
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--tied", action="store_false")
    parser.add_argument("--ae_scaled", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--online_training", action="store_true")
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--noise", type=float, default=0.0)
    
    args = parser.parse_args()
    
    return args