# HR-distance-number-of-shots-variance
Our goal is to determine the variance of HR distance with respect to number of shots.
Currently HR_last_param_var.py returns a list of HR distance neasurements.
Please edit the codebase appropraitely. 

## Example command to run VQE 
> python VQE_run.py --n_qbts 6 --shots 10000 --max_iter 10000 --n_layers 3 --output_dir tr1

To get more info about each hyperparameter:
> python VQE_run.py -h

## Example command to run HR distance measurement to get the variance
> python HR_last_param_var.py  --input_dir tr1 --shots 100 --num_HR 100

To get more info about each hyperparameter:
> python HR_last_param_var.py -h

(Note that the __--input_dir__ has to be the same as the __--output_dir__)

testing ssh key
