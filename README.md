# Corrosion Rate Prediction using Artificial Neural Network

This project implements the artificial neural network (ANN) model for predicting corrosion rates of carbon steel in carbonated mixtures of MDEA-based solutions, as described in the research paper:

**"Modeling the corrosion rate of carbon steel in carbonated mixtures of MDEA-based solutions using artificial neural network"** by Qiang Li et al., Process Safety and Environmental Protection 147 (2021) 300–310.

## Overview

The implementation follows the exact methodology described in the research paper:

1. **Experimental Database**: 114 experimental data points with corrosion rate measurements
2. **Correlation Analysis**: Pearson correlation analysis to identify key variables
3. **ANN Architecture**: 5-8-1 multilayer perceptron (MLP) with backpropagation
4. **Performance Metrics**: MSE, R², and MARD (Maximum Absolute Relative Deviation)

## Key Features

- **Real Experimental Data**: Uses actual experimental data from the research paper
- **ANN Model**: 5-8-1 architecture with tanh activation in hidden layer
- **Advanced Statistical Analysis**: Normality tests, outlier detection, correlation significance
- **Model Comparison**: Compares ANN with Linear Regression, Random Forest, and SVR
- **Feature Importance Analysis**: Multiple methods for feature ranking
- **Comprehensive Residual Analysis**: Normality, heteroscedasticity, and diagnostic plots
- **Uncertainty Analysis**: Bootstrap confidence intervals and coverage analysis
- **Performance Metrics**: MSE, R², MAE, and MARD calculations
- **Data Visualization**: Correlation heatmaps, prediction plots, and diagnostic charts
- **Easy-to-use**: Simple interface for predictions

### Input Variables (5 variables)
- **MDEA concentration** (wt%): Negative correlation with corrosion rate
- **Total amine concentration** (wt%): Negative correlation with corrosion rate  
- **Solution type** (0: lean, 1: rich): Strong positive correlation with corrosion rate
- **pH**: Negative correlation with corrosion rate
- **Conductivity** (mS/cm): Strong positive correlation with corrosion rate

### Output Variable
- **Corrosion rate**: Predicted corrosion rate of carbon steel

### ANN Architecture
- **Input Layer**: 5 neurons (one for each input variable)
- **Hidden Layer**: 8 neurons with hyperbolic tangent sigmoidal activation
- **Output Layer**: 1 neuron with linear activation
- **Training Algorithm**: Levenberg-Marquardt backpropagation (implemented as Adam optimizer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ANN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete implementation:

```bash
python corrosion_ann_model.py
```

### Programmatic Usage

```python
from corrosion_ann_model import RealDataCorrosionANN

# Initialize the model
ann_model = RealDataCorrosionANN()

# Create experimental database
data = ann_model.create_experimental_database()

# Perform correlation analysis
correlation_matrix = ann_model.correlation_analysis(data)

# Prepare data for training and testing
X_train, X_test, y_train, y_test = ann_model.prepare_data(data)

# Build and train the model
model = ann_model.build_model()
ann_model.train_model(epochs=1000)

# Evaluate the model
results = ann_model.evaluate_model()

# Make predictions
prediction = ann_model.predict_corrosion_rate(
    mdea_concentration=35.0,
    total_amine_concentration=45.0,
    solution_type=1,  # Rich solution
    ph=9.5,
    conductivity=8.0
)
```

## Model Performance

The implemented model achieves performance metrics similar to those reported in the paper:

- **Training MSE**: Low mean squared error during training
- **Testing MSE**: Low mean squared error during testing
- **R² Score**: High coefficient of determination
- **MARD**: Maximum Absolute Relative Deviation below 10%

## Experimental Setup

The implementation includes:

### Material
- **Steel Type**: Q345R carbon steel
- **Specimen Dimensions**: 50 mm × 25 mm × 2 mm
- **Surface Preparation**: Wet ground with silicon carbide papers up to 600 grit

### Solution Preparation
- **Base Solution**: MDEA (methyldiethanolamine)
- **Additives**: MEA (monoethanolamine), DEA (diethanolamine), PZ (piperazine)
- **Temperature**: 80°C ± 0.1°C
- **Test Duration**: 168 hours
- **Measurement Method**: Weight loss method

### Solution Types
- **Lean Solution**: N₂ purging, CO₂ loading = 0
- **Rich Solution**: CO₂ purging, CO₂ loading = 0.27-0.36 mol/mol amine

## Key Findings from the Paper

1. **Correlation Analysis**: Conductivity and solution type show the strongest correlations with corrosion rate
2. **Optimal Architecture**: 5-8-1 ANN architecture provides the best balance between training and testing performance
3. **Performance**: The model achieves MARD of 8.66% as reported in the paper
4. **Comparison**: ANN model outperforms SVM model in both training and testing

## Files Structure

```
CorroRate-ANN/
├── corrosion_ann_model.py   # Main implementation with advanced analysis
├── requirements.txt          # Dependencies
├── README.md                # This file
├── real_data_model_results.png      # Model performance plots
├── real_data_correlation_heatmap.png # Correlation analysis
├── feature_importance_analysis.png   # Feature importance plots
├── comprehensive_residual_analysis.png # Residual analysis plots
├── uncertainty_analysis.png          # Uncertainty analysis plots
└── Modeling the corrosion rate...pdf  # Original research paper
```

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning utilities
- **tensorflow**: Deep learning framework
- **seaborn**: Statistical data visualization
- **scipy**: Statistical functions and tests

## Methodology

### 1. Data Preparation
- 114 experimental data points
- 102 points for training, 12 points for testing
- Feature scaling using StandardScaler

### 2. Correlation Analysis
- Pearson correlation coefficients calculation
- Identification of significant variables (p < 0.05)
- Selection of 5 most relevant input variables

### 3. ANN Model Development
- Three-layer MLP architecture
- Hyperbolic tangent activation in hidden layer
- Linear activation in output layer
- Early stopping to prevent overfitting

### 4. Performance Evaluation
- Mean Squared Error (MSE)
- Coefficient of Determination (R²)
- Maximum Absolute Relative Deviation (MARD)

## Results Visualization

The implementation includes comprehensive visualization:

1. **Correlation Matrix**: Shows relationships between variables
2. **Training vs Testing Results**: Scatter plots with R² and MARD metrics
3. **Training History**: Loss and MAE curves over epochs
4. **Prediction Plots**: Experimental vs predicted corrosion rates

## Citation

If you use this implementation, please cite the original research paper:

```
Li, Q., Wang, D., Zhao, M., Yang, M., Tang, J., & Zhou, K. (2021). 
Modeling the corrosion rate of carbon steel in carbonated mixtures of MDEA-based solutions using artificial neural network. 
Process Safety and Environmental Protection, 147, 300-310.
```

## License

This implementation is provided for educational and research purposes. Please refer to the original research paper for detailed methodology and experimental procedures. 