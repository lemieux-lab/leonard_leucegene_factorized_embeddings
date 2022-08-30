# leonard_leucegene_factorized_embeddings

## Data 
To prepare the datasets, run:
```{bash}
python3 prepare_data.py
```
The files will be generated automatically into the "Data" folder.

Some specialized IO python packages are may be required, so the use of an external virtual environment is needed prior to loading the data. Access to the paths must be granted by the IRIC informatics platform server manager or accessed internally.

## Running experiments 
To run experiments in REPL for development (recommended), use a julia running IDE like vscode and execute "experiment_xx.jl" files. 
It is also possible to transfer executable julia files to other servers and running with: 
```{bash}
julia experiment_1.jl
```
*Note*: current Julia version is 1.8.0
