This is a demo file for end-to-end training of Deep Canonical Correlation Analysis (DCCA).

The code is written by Weiran Wang, one author of the paper "On deep multi-view representation learning (wang2015deep)".
The code is tested on the synthetic noisy MNIST dataset (described in 'wang2015deep').

Sometimes, the eigen-decomposition in tensorflow is not numerically stable, and it might report 'Nan' -- Please consider to use large minibatch.

If you use the code, please consider to cite the following two papers.

@inproceedings{wang2015deep,
  title={On deep multi-view representation learning},
  author={Wang, Weiran and Arora, Raman and Livescu, Karen and Bilmes, Jeff},
  booktitle={International Conference on Machine Learning},
  pages={1083--1092},
  year={2015}
}

@inproceedings{andrew2013deep,
  title={Deep canonical correlation analysis},
  author={Andrew, Galen and Arora, Raman and Bilmes, Jeff and Livescu, Karen},
  booktitle={International conference on machine learning},
  pages={1247--1255},
  year={2013}
}

