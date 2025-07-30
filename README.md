# Student Performance Prediction Through Lifestyle Analysis

A machine learning project that predicts academic performance using lifestyle factors rather than traditional academic metrics.

## Why This Project Matters

Most academic performance prediction focuses on obvious factors like previous grades and study hours. But what about the student who studies extensively but struggles due to sleep deprivation? Or the one whose performance drops because they're working part-time to support their family?

This project takes a different approach by examining how daily lifestyle choices impact academic success. We analyze factors like sleep patterns, financial stress, social media usage, and family dynamics to build a more complete picture of student performance.

## The Problem

Traditional performance prediction misses crucial lifestyle factors that significantly impact academic success. Students facing financial stress, irregular sleep, or excessive family pressure often struggle despite their academic potential. Early identification of these patterns can help universities provide targeted support before problems become severe.

## Our Approach

We predict performance categories (Excellent/Good/Average/Poor) using lifestyle and behavioral data including:

- **Sleep and Health**: Hours of sleep, exercise frequency, overall health status
- **Social Factors**: Social media usage, relationships, extracurricular activities  
- **Financial Reality**: Income level, part-time work, financial stress
- **Family Dynamics**: Support level, pressure, educational background
- **Living Situation**: Hostel vs home, travel time, accommodation type
- **Study Environment**: Internet access, library usage, group study habits

## Technical Implementation

### Algorithms Used

**Random Forest**: Handles mixed data types well and provides feature importance rankings. Shows which lifestyle factors matter most for different students.

**Naive Bayes**: Excellent for probabilistic classification with lifestyle data. Works effectively even with smaller datasets.

**Decision Trees**: Creates interpretable rules that can guide practical interventions. Generates clear "if-then" recommendations.

**Ensemble Method**: Combines all models using voting classifier to improve prediction reliability and reduce individual model biases.

### Dataset Sources

- UCI Student Performance Dataset (mathematics and Portuguese language courses)
- Student lifestyle and sleep pattern datasets from Kaggle
- Custom demonstration data incorporating Pakistani university context

## Getting Started

### Requirements

```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

### Installation

```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
python student_performance_classifier.py
```

### Usage

The script automatically handles data loading, preprocessing, model training, and evaluation. It provides:

1. **Data exploration** with visualizations showing lifestyle vs performance relationships
2. **Model training** with cross-validation and performance comparison
3. **Results analysis** including feature importance and prediction accuracy
4. **Actionable insights** for students and educational institutions

## Results and Insights

### Key Findings

**Sleep Quality Dominates**: Sleep hours often predict performance better than study hours. Well-rested students with moderate study time frequently outperform exhausted students who study extensively.

**Financial Stress Impact**: Students experiencing financial pressure show distinct performance patterns regardless of their academic ability.

**Social Media Balance**: Moderate usage correlates with better performance (staying connected, accessing resources), while excessive use clearly impacts grades negatively.

**Family Dynamics**: Relationship quality and pressure levels significantly affect academic outcomes, particularly in cultures with high family expectations.

### Model Performance

- Accuracy: 75-85% depending on dataset quality
- Cross-validation stability: ±3-5%
- Clear feature importance rankings identify intervention priorities

## Applications

### For Students
- Personal pattern recognition and lifestyle optimization
- Early warning system for potential academic difficulties
- Data-driven recommendations for habit changes

### For Universities  
- Early identification of at-risk students
- Targeted support service allocation
- Evidence-based policy development for student welfare

### For Researchers
- Cultural factor analysis in academic performance
- Quantified lifestyle impact assessment
- Framework for intervention design

## Contributing

We welcome contributions in several areas:

- **Data Collection**: Additional datasets from diverse educational contexts
- **Model Enhancement**: Algorithm improvements and feature engineering
- **Visualization**: Better charts and interactive dashboards  
- **Documentation**: Clearer explanations and use case examples

## Ethical Considerations

This tool should support student success, not create barriers. Important principles:

- Use predictions for intervention, not discrimination
- Respect student privacy and data protection requirements
- Remember that predictions are probabilities, not certainties
- Focus on helping students rather than labeling them

## Limitations

- Model accuracy depends entirely on training data quality
- Cultural factors may not transfer across different regions
- Individual circumstances can override general patterns
- Correlation in data doesn't guarantee causation

## Technical Details

The implementation uses scikit-learn for machine learning, pandas for data manipulation, and matplotlib/seaborn for visualization. The code is designed to be educational, with extensive comments explaining each step of the analysis process.

Cross-validation ensures model reliability, while ensemble methods combine multiple algorithms to improve prediction accuracy. Feature importance analysis identifies which lifestyle factors have the strongest relationship with academic performance.

## License

MIT License - see LICENSE file for details.

---

**Note**: This project aims to help educational institutions better support their students by understanding the relationship between lifestyle choices and academic success. All analysis should be conducted with appropriate ethical oversight and student consent.
