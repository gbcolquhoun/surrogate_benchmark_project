# main.py
from evo_search.searcher import evo_search
import os

# Set the working directory to the script's directory
print(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    '''
    Start point of project
    '''
    
    print("Starting script main.py...")
    design_space = "configs/hat_design_space.yaml"
    search_params = "configs/evo_search_params.yaml"
    print(" initializing evo_search")
    search = evo_search(design_space, search_params)
    #search.run_evo_search()
    #search.run_exhaustive_search()
    summary_df = search.run_validation_study()
    print(summary_df.head())


if __name__ == "__main__":
    main()