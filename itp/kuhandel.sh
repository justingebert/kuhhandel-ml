#! /bin/bash
file='/home/jnn.aurich/kuhhandel-ml/itp/train_self.py' # adjust to run other script

# echo "first  command line parameter: "$1
# echo "second command line parameter: "$2

# activate python environment, make a wrapper for it (otherwise command line arguments are passed to it)
wrappersource() {
    source /home/jnn.aurich/kuhhandel-venv/bin/activate
}
wrappersource
# run script with command line arguments (relevant for job arrays) $1=Cluster, $2=Process 
python3 $file #$1 $2
exit