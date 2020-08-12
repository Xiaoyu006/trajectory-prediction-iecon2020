# trajectory-prediction-iecon2020

This repo contains the code for the paper published in IECON-2020.

## Data preprocessing

Run the following commands in order to preprocess the data. You may need to download raw NGSIM trajectories first and make sure the NGSIM file is at the proper direction and with a proper name ,e.g.,`../../trajectories-0805am-0820am.csv`.

`python data_pieces_selection.py`

`python combine_data.py`

`python random_split_data.py`

After that you can start to train the model.

## Training

