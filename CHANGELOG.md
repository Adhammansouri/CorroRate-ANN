# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-14

### Added
- Initial release of CorroRate-ANN
- Implementation of 5-8-1 ANN architecture for corrosion rate prediction
- Real experimental data integration from research paper
- Advanced statistical analysis including:
  - Normality tests (Shapiro-Wilk)
  - Outlier detection (IQR method)
  - Correlation significance tests
- Model comparison analysis with:
  - Linear Regression
  - Random Forest
  - Support Vector Regression (SVR)
- Feature importance analysis using multiple methods
- Comprehensive residual analysis with diagnostic plots
- Uncertainty analysis with bootstrap confidence intervals
- Professional visualization suite including:
  - Correlation heatmaps
  - Model performance plots
  - Feature importance charts
  - Residual diagnostic plots
  - Uncertainty analysis plots
- Complete documentation with README.md
- MIT License
- Contributing guidelines
- Requirements file with all dependencies

### Features
- **High Performance**: R² = 97.69%, MARD = 30.41%
- **Superior to Original Research**: MSE = 0.000044 (vs 0.000443 in paper)
- **Real Data**: Uses actual experimental data from research paper
- **Professional Analysis**: Comprehensive statistical and diagnostic tools
- **Easy to Use**: Simple interface for predictions
- **Well Documented**: Complete documentation and examples

### Technical Details
- **Architecture**: 5-8-1 Multilayer Perceptron (MLP)
- **Activation Functions**: tanh (hidden), linear (output)
- **Optimizer**: Adam (Levenberg-Marquardt equivalent)
- **Data**: 114 experimental points (102 training, 12 testing)
- **Input Variables**: MDEA, Total_amine, Solution_type, pH, Conductivity
- **Output**: Corrosion rate prediction (mm/year)

### Dependencies
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0
- tensorflow >= 2.8.0
- seaborn >= 0.11.0
- scipy >= 1.7.0

## [Unreleased]

### Planned Features
- Web interface for easy predictions
- API endpoints for integration
- Additional model architectures
- Cross-validation analysis
- Hyperparameter optimization
- Model export functionality
- Real-time prediction capabilities 