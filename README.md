# gpt-clone
A clone repo of nano-gpt with personalized touch

## Docker Dev Environment
to run code inside docker dev environment, first build docker image <br>
```make build-torch-image``` <br>
next, run docker by mounting this directory into it and getting access via bash
```run-my-torch ```


### Data Download
to download the shakespeare data from inside docker <br>
```./data/download_shakespeare.sh``` <br> 
you may need to chmod the file
