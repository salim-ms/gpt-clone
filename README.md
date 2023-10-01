# gpt-clone
A clone repo of nano-gpt with personalized touch

## Docker Dev Environment
to run code inside docker dev environment, first build docker image <br>
```make build-torch-image``` <br>
next, run docker by mounting this directory into it and getting access via bash <br>
```run-my-torch ``` <br>
``` export PYTHONPATH=${PYTHONPATH}:$(pwd)```


### Data Download
to download the shakespeare data from inside docker <br>
```./data/download_shakespeare.sh``` <br> 
you may need to chmod the file

##### Saved Models will be stored under model_outputs/<model name>/model_saved.pth


# train bigram
```python train.py --config=bigram.ini```

# sample bigram model
```python sample.py --config=bigram.ini --max_tokens=500```


# train baby gpt with Shakespeare and char level
```python train.py --config=baby_gpt_char.ini```
# sample baby gpt
```python sample.py --config=baby_gpt_char.ini --max_tokens=500```

## Token Based Baby GPT
```
to train
python train.py --config=baby_gpt_tokens.ini
to sample
python sample.py --config=baby_gpt_tokens.ini

```

Output sample
```
eria ing i' the general dishonour of peace 
 much chest Whereto, his name summer store. 
 
 ROMEO: 
 Go, he, more than robs my Corioli is from all pieces. 
 
 JULIET: 
 What day defect ; until what's thee? 
 He is not on him to death's gentle Hastings speak: 
 The royalties of all able to the moon, 
 And watch'd such contempt, he's rid but near up me. 
 
 YORK: 
 And never so, it must become the common bastard kind, 
 Take her more heavier. 
 
 DERBY: 
 And, sir, he's mother, will, good they have gi' councils 
 And stolen are sure my mind for lost; 
 Methinks I will go turn now. 
 
 SICINIUS: 
 Thanks, noble brother? 
 
 SICINIUS: 
 Thou willingness Far school-maids prince and lie along? 
 
 Lord: 
 Be singly withal: Therefore be, the succor and for 
 doit. Well, what boy? 
 
 PAULINA: 
 By any that 
 That usurp'd: snow Our cheerful prancing to committed you thanks. 
 But thou canst draw thee for him, I love 
 Have I, to do not time how to her husband, 
 That will

```