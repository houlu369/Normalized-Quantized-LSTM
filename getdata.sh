echo "=== Acquiring datasets ==="
echo "---"
mkdir -p save

mkdir -p data
cd data




echo "- Downloading Penn Treebank (PTB)"
wget --quiet --continue http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
tar -xzf simple-examples.tgz

mkdir -p penn
cd penn
mv ../simple-examples/data/ptb.train.txt train.txt
mv ../simple-examples/data/ptb.test.txt test.txt
mv ../simple-examples/data/ptb.valid.txt valid.txt
cd ..

echo "- Downloading Penn Treebank (Character)"
mkdir -p pennchar
cd pennchar
mv ../simple-examples/data/ptb.char.train.txt train.txt
mv ../simple-examples/data/ptb.char.test.txt test.txt
mv ../simple-examples/data/ptb.char.valid.txt valid.txt
cd ..
rm -rf simple-examples/
rm  simple-examples.tgz

mkdir -p  warpeace
cd warpeace
wget https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt
python pre_warpeace.py
rm warpeace_input.txt
cd ..

mkdir  -p  text8
cd text8
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
python pre_text8.py
rm text8.zip
rm text8
cd ..

cd ..
echo "---"
echo "Happy language modeling :)"
