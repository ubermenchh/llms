# Lil Language Models

Parameters Used:
```
batch_size = 32
epochs = 1000
learning_rate = 3e-4
log_interval = 100
n_embd = 384
n_heads = 8
n_layers = 4
dropout = 0.2
context_window = 16
d_model = 512
```

## 1. GPT Model
All thanks to [Karpathy](https://github.com/karpathy) senpai.

To run the model:

```python

!python train.py --file="shakespeare.txt" --model="gpt"
```

First Run produced a Validation Loss of 1.9375 with some gibberish type of shakespeare. [Long way to go ;)]

Shakespeare when he was 5 i think:
```
quen redwelor iff
gauntenged:
that?

king learl, therem the boy stay: sep;

ove that be to ive suler'd,
wo throthing and thy kin'd well ceon'des benter' bettred, this you areat:--
love to vianterty
the our
when finly and lond!

pourporse come inneclary the not.
evel:
and tay thefrother.
ilt;
the pruplay heaquests---
feary on in sawen it elome stitant naight
then dough!

seconine:
the sikes rosent.

juster:
the nom well be undek one it senay that should'd ore they 'not do thee, if by hey?

nuve toe,
love thou pray moke nay,
but that i gruce.
and twele the firsent beittes.
swarew:'s we land deves elvole then the deten,
i lord do a you, my you it porse play wase
and me you'll sward,
a make vildrations you, eaven
i, my henromay, the visting dotomeran:
nowat to urt glea, why aurn,
challves and out dear.

quen:
nerinnausand have cleabut broubans, of our crefoves the be wen wine cre thicher.
'use, get and my a hencate?
beala:
lay by, ehtreghngs of cept?
own: hepthallone she menthat unemon winst a noow thee clomed t
```


## 2. LLaMa
Thanks to this [repo](https://github.com/bkitano/llama-from-scratch).

To run the model:
```
!python train.py --file="shakespeare.txt" --model="llama"
```

First run produced a Validation Loss of 0.1472 (not bad but wait), with some shakepeare i could've have produced whie mumbling in my sleep.

Shakespeare if he was 3 i think:
```
ayyyyy?


unerhy:
or apu, a dilv your st,
wemfay cimrerss:
my mus suringd sall m-
friche!

dowis; shimy noouwx gry lie o'gs xuucelar quees eunts crousa yove sten:
, res   nous satd;
ing
whend shivunthig, hing ot she def gteuvent,
him this bund potrs;
tioctgci, hes with yout wercus.

andus:
king monn atsy sewsthr guy
buig-hen.

veell kin:
ti 's thome, thah curd bund, wins vengst
 medssy, phofingure?
ing, young use as te aved yous:
pule pamy,
thee an kend itss pre,
and and nome snee bcre
warices tricirtt gyn them,
rund me tin pone ot have te ne  dungeme; sppard
veked,
av:
s in, te cacus cave ifn?
cruts quis:
iprce, rifp niss ba?
avin win:
pe dehum stwhend feir diak stves! mingeind ing cest ui, thevounck hat your grathptoland avaveit di bulle tnut besm 'tracks
as fokivechuking,
in' veack's thin'll:
the thoun wich ave thesurksbuns, insth thes,
fim suse lef wopr lount cunls me, shave; mand eumve
se, nit with:
the 'y.
bus s,
thirtptland fromer ojund omallyaus thol! bush:
such,, nolis, rce, yor.

blis trilickingicie
```


