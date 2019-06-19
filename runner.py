import sim_all
import sys






if __name__=="__main__":
    
    ##parse command line arguments
    
    population = sys.argv[1]
    update_time = sys.argv[2]
    study_length = sys.argv[3]
    start_index = sys.argv[4]
    end_index = sys.argv[5]
    case =sys.argv[6]
    train_type =sys.argv[7]
    algtype =sys.argv[8]
    
    root = 'distributions_rl4rl/'
    write_directory = 'results/'
    sim_all.run_many(algtype,[case],int(start_index),int(end_index),int(update_time),root,write_directory,train_type)


