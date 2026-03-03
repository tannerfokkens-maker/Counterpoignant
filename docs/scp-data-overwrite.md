# SCP Prepared Data Override

Run from your local machine after `prepare-data` completes:

```bash
cd /Users/tannerfokkens/Documents/2pt-bach_update

# ensure remote data dir exists
ssh ubuntu@209.20.156.114 'mkdir -p ~/Counterpoignant/data'

# overwrite remote prepared data files
scp -C \
  data/tokenizer.json \
  data/sequences.json \
  data/piece_ids.json \
  data/corpus_stats.json \
  data/mode.json \
  ubuntu@209.20.156.114:~/Counterpoignant/data/
```

Optional verify on server:

```bash
ssh ubuntu@209.20.156.114 'cd ~/Counterpoignant && ls -lh data/{tokenizer.json,sequences.json,piece_ids.json,corpus_stats.json,mode.json}'
```
