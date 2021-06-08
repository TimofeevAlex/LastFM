# LastFM

This project was carried out as part of the Machine Learning for Behavioural Data course at EPFL in the spring 2020 semester. The aim of the project was to use the LastFM dataset containing information on users' listening habits, as well as demographic information, to recommend new artists for users. 

## Using the code

In this project, we use the `TensorFlow` library to construct, fit, and evaluate models, and matplotlib to visualize results.

###  Data

The original data contains two datasets:
 - `usersha1-profile` containing demographic data, with columns `user_email`, `gender`, `age`, `country`, `signup`
 - `usersha1-artmbid-artname-plays` containing behavioural data, with columns `user_email`, `artist_id`, `artist_name`, `plays`

The cleaned and preprocessed data can be generated by following the instructions in the `preprocessing.ipynb` notebook. In order to upload the original dataset, the cells with the title "Uploading the original dataset" must be uncommented. It is also possible to find the prepared data at the following link: https://drive.google.com/drive/folders/14izEunqUyASA-fkS_EqrBcWMpHI1pXxQ

#### Preprocessing:

The following transformations are performed on the data to prepare it for use:
 - dataset is reduced to information about users in USA
 - `signup` column is converted into datetime type
 - user IDs are converted to numbers
 - unrealistic ages are changed to NaN
 - gender, country, and age are one-hot encoded
 - features `year`, `month`, `weekday` and `day` are extracted from sign-up date
 - samples with missing `artist_name` are dropped
 - samples with missing `artist_id` are dropped
 - implicit feedback is transformed to explicit (plays to ratings)

The processed datasets are saved in the folder `lastfm-dataset-360K` with file names `behav-360k-processed.csv` and `demo-360k-processed.csv`

Finally, the data is split into train, validation, and test datasets, with 10% of the whole set being used for testing, and 10% of the training set used for validation. The function `train_test_split` from the `train_test.py` module is used to split the data.

### Training

The following models and settings are implemented and evaluated:
- Baseline Model, with and without cold start
- User-User Neighborhood Model
- Latent Factor Model, with different proportions of negative samples
- Neural Matrix Factorization Model, with and without cold start, and with different proportions of negative samples

For training, labels are converted to `0` if there is no interaction between the given user-artist pair, and `1` otherwise. The ratings calculated during preprocessing are used as weights for thr Weighted Binary Cross Entropy when training LFM and NeuMF.

The following notebooks should be run to train and evaluate the models:
- `baseline_model.ipynb`
- `user_neighborhood_model.ipynb`
- `Deep_latent_FM_main.ipynb`
- `latent_factor_model.ipynb`

Each notebook corresponds to a model, and prepares data and trains the model in all necessary settings (different number of negative samples, cold start, etc).

### Environment

The project has been developed with python `3.7.10`. The full requirements are stated in the `requirements.txt` file.
All necessary imports are done at the beginning of each notebook.














