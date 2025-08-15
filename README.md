# Digital_Chemistry
Prediction of Absorption Maxima of Organic Dyes Using Molecular Descriptors and Machine Learning

## Idea: 
Build a model to predict the absorption maxima of organic chromophores based on their molecular structures and solvent environments.

## Motivation:
- In dye chemistry, absorption maxima is critical for applications in solar cells, bioimaging, and photodynamic therapy.
- Experimental measurement is slow and costly.
- Goal: Predict λmax from structure to speed up screening.

## Methods:
- Feature Engineering:
. RDKit descriptors
. Morgan Fingerprints
. One-hot encoding 

- Models tested:
. Dummy Regressor (baseline)
. Ridge Regression
. Random Forest Regressor

- Evaluation: GroupKFold cross-validation
- Metrics: RMSE, MAE, R-square.

## Results:
- Predicted vs True λmax scatter plots for Ridge & RF.
- Residual histograms: RF residuals centered near zero with narrower spread.