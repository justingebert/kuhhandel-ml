#! /bin/bash
# Point to the unified script in the parent directory
file='/home/jnn.aurich/kuhhandel-ml/rl/train/train_selfplay.py' 

# echo "--itp"$1
# echo "second command line parameter: "$2

# activate python environment, make a wrapper for it (otherwise command line arguments are passed to it)
wrappersource() {
    source /home/jnn.aurich/ml/bin/activate
}
wrappersource
# run script with command line arguments (relevant for job arrays) $1=Cluster, $2=Process 
python3 $file --itp --hyperparams high_range --suffix _high_range
exit
