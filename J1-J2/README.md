# HR distance variance measurement for J1-J2 model
Our goal is to determine the variance of HR distance with respect to number of shots.
Currently HR_J1_J2_last_param_var.py returns a list of HR distance neasurements.
Please edit the codebase appropraitely. 

## Example command to run VQE 
> python VQE_J1_J2.py --m 3 --n 2 --J1 0.5 --J2 0.2 --ansatz_type ALA --shots 10000 --max_iter 10000 --n_layers 3 --output_dir tr1

To get more info about each hyperparameter:
> python VQE_J1_J2.py -h

## Example command to run HR distance measurement to get the variance
> python HR_J1_J2_last_param_var.py  --input_dir tr1 --shots 1000 --num_HR 10

To get more info about each hyperparameter:
> python HR_J1_J2_last_param_var.py -h

(Note that the __--input_dir__ has to be the same as the __--output_dir__)
