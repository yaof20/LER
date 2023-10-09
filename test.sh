train_data=harm-wo-test
encoder=shared
pooling=avg
backbone=bert-tiny

attention=dot-attention
is_weighted=non-weighted

for config in `ls ./config/$train_data/$encoder/$pooling/$backbone/$attention/$is_weighted`
  do
    python test.py --config ./config/$train_data/$encoder/$pooling/$backbone/$attention/$is_weighted/$config
  done