# Spaceship Titanic: Data Science Challenge

## Overview
Welcome to the year 2912! Your data science skills are needed to solve a cosmic mystery. We’ve received a transmission from four light-years away, and things aren’t looking good.

The Spaceship Titanic, an interstellar passenger liner, was launched a month ago, carrying nearly 13,000 passengers on its maiden voyage to three newly habitable exoplanets. While rounding Alpha Centauri on its way to the first destination—the scorching 55 Cancri E—the Spaceship Titanic encountered a hidden spacetime anomaly within a dust cloud. Sadly, it met a fate similar to its namesake from 1,000 years ago. Although the ship remained intact, nearly half of the passengers were transported to an alternate dimension!

To assist rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

## Objective
The goal of this project is to build a predictive model that accurately identifies which passengers were affected by the spacetime anomaly. The results will aid in the rescue operation and help bring the passengers back safely.

## Dataset
The dataset consists of records retrieved from the ship's damaged computer system, containing information on each passenger. Each entry includes various attributes that may help determine if a passenger was transported to an alternate dimension.
- Percorre os arquivos no local indicado
  - Verifica se existem pastas para a extensão, caso não, cria a pasta
  - Move os arquivos com extensões para as pastas correspondentes
  > O programa não move arquivos .exe para evitar erros com o executável devido à caminho incorreto
- Caso aconteça algum erro, exibe no terminal

## Requirements
- ```Pandas```
- ```Seaborn or Matplotlib```
- ```Tensorflow```
  > Used ```Scikit-learn``` because ```tensorflow-decision-forests``` does not currently have support for python 3.12

## Instalation
- Clone this repository:
```
git clone https://github.com/victorsimasdev/spaceship_titanic
```
- Install the required packages:
```
pip install -r requirements.txt
```

## Aproach
- Data Preprocessing:
  - Removed columns ```Name``` and ```Cabin``` which were not relevant for the prediction.
  - Handled missing values:
    - Categorical columns like ```HomePlanet```, ```CryoSleep```, ```Destination```, and ```VIP``` were filled with their most frequent values.
    - Numerical columns related to onboard services were filles with zeros, assuming missing data as no service used.
    - ```Age``` was filled with the median age.
  - Converted categorical columns into dummy variables using one-hot encoding.
- Modeling:
  - Used **Random Forest Classifier** for prediction.
  - Split the dataset into training and validation sets (80/20 split).
  - Trained the model and evaluated performance using accuracy, precision, recall, and F1-score.
  - Visualized the classification report and confusion matrix to assess model performance.
- Visualization:
  - Protted age distribution based on transported status.
  - Displayed a bar chart of precision, recall, and F1-score for each class.
  - Visualized the confusion matrix to analyze prediction accuracy.
- Test Data Preparation and Prediction:
  - Followed similar preprocessing steps for the test data.
  - Ensured the test data columns matched the training data.
  - Predicted the transported status for the test data and saved the results in a CSV file name ```submission.csv```.
## Usage
To train the model and generate prediction:
```
python spaceship_titanic.py
```

## Results
The final model achieved an accuracy of **78%**. It provides a strong basis for predicting transported passengers and assisting in their rescue.

### Classification Report
![Classification Report](https://github.com/user-attachments/assets/2bd90729-351a-45f8-951c-724e74aaba61)

### Age Distribution of Transported Passangers
![Age Distribution](https://github.com/user-attachments/assets/601f478d-438d-47c5-83c9-00030086e281)

### Confusion Matrix
![Confusion Matrix](https://github.com/user-attachments/assets/413d1268-a870-4979-b972-60a6c7df61c8)

## Future Work
- Future Engineering:
  - Additional derived features could improve the model.
- Ensemble Methods:
  - Trying ensemble techniques for portentially better performance.
- Deep Learning:
  - Exploring neural networks to futher enhance the predictive power.
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
