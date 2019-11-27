
## Prepare your envs
1. pip install -r requirements.txt


## Language modeling


### Prepare your data
1. run `bash getdata.sh` to prepare data.

### Run an example
1. Character-level language modeling 
```
python main_lm.py  --data data/pennchar --batch_size 64 --emsize 512 --nlayers 1 --nhid 512 \
--bptt 100 --lr 0.002  --epochs 200  --dropouto 0. --dropouth 0. --dropouti 0. \
--dropoute 0. --alpha 0. --beta 0. --wdecay 0. --save pennchar \
 --optimizer adam --clip 5  --char --norm batch  --method binary_connect  --shared
```
```
python main_lm.py  --data data/pennchar --batch_size 64 --emsize 512 --nlayers 1 --nhid 512 \
--bptt 100 --lr 0.002  --epochs 200  --dropouto 0. --dropouth 0. --dropouti 0. \
--dropoute 0. --alpha 0. --beta 0. --wdecay 0. --save pennchar \
 --optimizer adam --clip 5  --char --norm weight  --method binary_connect 
```

2. Word-level language modeling
```
python main_lm.py  --data data/penn --batch_size 20 --emsize 300 --nlayers 1 --nhid 300 \
--bptt 35 --lr 20 --lr_decay 4 --epochs 120 --dropouto 0.5 --dropouth 0.0 --dropouti 0.5 \
--dropoute 0. --alpha 0. --beta 0. --wdecay 0. --save PTB10bianry  \
  --clip 0.25  --norm batch --method binary_connect --shared
```

##  Sequential mnist
### Run an example
```
python main_mnist.py --norm weight --method twn
```
