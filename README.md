# Domain-shift Conditioning using Adaptable Filtering via Hierarchical Embeddings for Robust Chinese Spell Check

Spell check for Chinese poses unresolved problems due to the large number of characters, the sparse distribution of errors, and the dearth of resources with sufficient coverage of shifting error domains.
Hierarchical Embedding Adaptable filter (HeadFilt) is a scalable adaptable filter that exploits [Hierarchical character embeddings](https://github.com/mnhng/hier-char-emb) to improve Chinese spell check accuracy.
HeadFilt (1) obviates the need for handcrafted resources covering different error domains and (2) resolves sparsity problems related to infrequent errors.
Both Simplified and Traditional Chinese input are supported.

If you use this code, please cite as appropriate:

```
@article{nguyen2021domain,
  title={Domain-shift Conditioning using Adaptable Filtering via Hierarchical Embeddings for Robust Chinese Spell Check},
  author={Nguyen, Minh and Ngo, Gia H and Chen, Nancy F},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2021}
  publisher={IEEE}
}
```

## Requirements

Python and PyTorch are required for the current codebase.
To setup the required environment

1. Install Anaconda
2. Run `conda env create -f env.yml -n spell`


## Examples:

### Data

Processed data from the Chinese Spelling Check (CSC) Task in 2013, 2014, and 2015 (both Traditional and Simplified scripts) are included.

HeadFilt model weights can be downloaded from: 

+ For Simplified text: [hierEmbWeights\_d512\_m0.4\_simp\_ft.pt](https://drive.google.com/file/d/1mu1ivyFVd7DIT4tiYU1x6iJ5CVm_dW4Y/view?usp=sharing)
+ For CSC 2013 Traditional text: [hierEmbWeights\_d512\_m0.4\_trad\_ft13.pt](https://drive.google.com/file/d/14vuGMXIFHKiZVfOHf8-BYkHzJFAPH-5g/view?usp=sharing)
+ For CSC 2014 Traditional text: [hierEmbWeights\_d512\_m0.4\_trad\_ft14.pt](https://drive.google.com/file/d/1OdmnzSDJo2wPsQApsYc56b4yU_yJMGax/view?usp=sharing)
+ For CSC 2015 Traditional text: [hierEmbWeights\_d512\_m0.4\_trad\_ft15.pt](https://drive.google.com/file/d/1tVgZjJguUwPlMeRHkKS6g3xZ4uaYzxVS/view?usp=sharing)

### Example - Spelling check for Traditional text

#### 0. Activate environment
```bash
source activate spell
```

#### 1. Train base classifier on CSC 2015 (Traditional text)
```bash
./csc_base.py --config conf/15.trad.json --seed 1 --outpath out/csc15.trad.s1
```

#### 2. Use fixed filtering to improve prediction on CSC 2015 (Traditional text)
```bash
./csc_fixedfilt.py --config conf/15.trad.json --checkpoint out/csc15.trad.s1/
```

#### 3a. Estimate distance factor of HeadFilt (you should get around 0.09843067824840546)
```bash
./estimateNegDist.py --margin 0.4 -c hierEmbWeights_d512_m0.4_trad_ft15.pt
```

#### 3b. Use HeadFilt to improve prediction on CSC 2015 (Traditional text)
```bash
./csc_headfilt.py --config conf/15.trad.json --checkpoint out/csc15.trad.s1/ --margin 0.4 \
    --emb_weights hierEmbWeights_d512_m0.4_trad_ft15.pt --factor 0.09843067824840546
```

### Example - Spelling check for Simplified text

#### 0. Activate environment
```bash
source activate spell
```

#### 1. Train base classifier on CSC 2015 (Simplified text)
```bash
./csc_base.py --config conf/15.simp.json --seed 1 --outpath out/csc15.simp.s1 --max_train_epoch 3
```

#### 2. Use fixed filtering to improve prediction on CSC 2015 (Simplified text)
```bash
./csc_fixedfilt.py --config conf/15.simp.json --checkpoint out/csc15.simp.s1/
```

#### 3a. Estimate distance factor of HeadFilt (you should get around 0.07925131916999817)
```bash
./estimateNegDist.py --margin 0.4 -c hierEmbWeights_d512_m0.4_simp_ft.pt --simplified
```

#### 3b. Use HeadFilt to improve prediction on CSC 2015 (Simplified text)
```bash
./csc_headfilt.py --config conf/15.simp.json --checkpoint out/csc15.simp.s1/ --margin 0.4 \
    --emb_weights hierEmbWeights_d512_m0.4_simp_ft.pt --factor 0.07925131916999817
```

### Example - Training HeadFilt from scratch (Traditional text)

#### 1. Imitation training
```bash
./imitation.py --lr 3e-3 --batch_size 500 -t 150000 --dim 512 --margin .4 --out hierEmbWeights_d512_m0.4_trad.pt
```

#### 2. Adapting to CSC 2015 (Traditional text)
```bash
./adapt.py --config conf/15.trad.json --lr 3e-3 --batch_size 500 -t 50000 --dim 512 --margin 0.4 -c hierEmbWeights_d512_m0.4_trad.pt \
    --out hierEmbWeights_d512_m0.4_trad_ft15.pt
```


### Links to original data

+ [SIGHAN 2013 Bake-off: Chinese Spelling Check Task](http://ir.itc.ntnu.edu.tw/lre/sighan7csc.html)
+ [CLP 2014 Bake-off: Chinese Spelling Check Task](http://ir.itc.ntnu.edu.tw/lre/clp14csc.html)
+ [SIGHAN 2015 Bake-off: Chinese Spelling Check Task](http://ir.itc.ntnu.edu.tw/lre/sighan8csc.html)
+ [Automatic-Corpus-Generation](https://github.com/wdimmy/Automatic-Corpus-Generation)
+ [SpellGCN](https://github.com/ACL2020SpellGCN/SpellGCN)
