Author: Dainis Boumber

# tf-dann-py3

Tensorflow-gpu (1.0.1, Window, py3) implementation of Domain Adversarial Neural Network.
Domain Adversarial Neural Network was published by Ajakan et al.  in  https://arxiv.org/abs/1412.4446

Modified from [jaejun-yoo](https://github.com/jaejun-yoo/tf-dann-py35)'s github

This DANN is suitable for non-image, classical ML data.

The code has been refactored and simplified.

Usage from IDE:

python DANN.py will run default supernova dataset. You can change that to mars dataset by editing main()

Usage from shell:

$ python DANN.py source target

The datasets are in the ./data/ directory of the project, and are assumed to be csv files with no header row.
in the above case, we would have ./data/source.csv and ./data/target.csv

