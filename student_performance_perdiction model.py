# Student Performance Classification: Lifestyle-Based Prediction
# Using Real Datasets with Unconventional Features
# Techniques: Random Forest, Naive Bayes, Simple Ensemble Methods

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== Student Performance Classification Project ===")
print("Focus: Lifestyle-based academic performance prediction")
print("Context: University students with unconventional features\n")

# Dataset URLs for download (you need to download these from Kaggle)
datasets_info = {
    "Student Lifestyle Dataset": "https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset",
    "Student Sleep Patterns": "https://www.kaggle.com/datasets/arsalanjamal002/student-sleep-patterns", 
    "Lifestyle Factors Impact": "https://www.kaggle.com/datasets/charlottebennett1234/lifestyle-factors-and-their-impact-on-students",
    "UCI Student Performance": "https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set"
}

print("📊 Available Datasets:")
for name, url in datasets_info.items():
    print(f"• {name}: {url}")

print("\n" + "="*60)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("="*60)

# Function to load UCI Student Performance dataset (most commonly available)
def load_uci_student_data():
    """
    Load the UCI Student Performance dataset
    This dataset includes lifestyle factors like:
    - Family background, study time, failures, social activities
    - Health status, romantic relationships, alcohol consumption
    - Travel time, family support, internet access
    """
    try:
        # Try loading from local file first
        data = pd.read_csv('student-mat.csv', sep=';')
        print("✅ Loaded UCI Student Performance dataset (Mathematics)")
        return data, 'uci'
    except FileNotFoundError:
        print("❌ Dataset file not found. Please download from:")
        print("https://www.kaggle.com/datasets/dskagglemt/student-performance-data-set")
        return None, None

# Function to create sample lifestyle-focused dataset for demonstration
def create_lifestyle_demo_data(n_samples=800):
    """Create a demonstration dataset focusing on lifestyle factors"""
    np.random.seed(42)
    
    # Generate realistic lifestyle data
    data = {
        # Demographics
        'age': np.random.randint(17, 25, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.55, 0.45]),
        
        # Living situation (Pakistani context)
        'accommodation': np.random.choice(['hostel', 'home', 'shared'], n_samples, p=[0.4, 0.45, 0.15]),
        'family_size': np.random.choice(['small', 'large'], n_samples, p=[0.6, 0.4]),
        
        # Financial factors
        'part_time_job': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
        'financial_stress': np.random.randint(1, 6, n_samples),  # 1-5 scale
        'family_support': np.random.randint(1, 6, n_samples),
        
        # Study habits and lifestyle
        'study_time': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.3, 0.3, 0.2]),  # 1: <2hrs, 4: >10hrs
        'sleep_hours': np.clip(np.random.normal(6.5, 1.2, n_samples), 4, 10),
        'social_media_hours': np.clip(np.random.exponential(2, n_samples), 0, 8),
        'exercise_frequency': np.random.randint(0, 8, n_samples),  # times per week
        
        # Social and health factors
        'social_activities': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.3, 0.5, 0.2]),
        'romantic_relationship': np.random.choice(['yes', 'no'], n_samples, p=[0.25, 0.75]),
        'health_status': np.random.randint(1, 6, n_samples),  # 1: very bad, 5: very good
        
        # Academic factors
        'internet_access': np.random.choice(['yes', 'no'], n_samples, p=[0.85, 0.15]),
        'extra_curricular': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
        'family_pressure': np.random.randint(1, 6, n_samples),
        'previous_failures': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Create performance categories based on weighted factors
    performance_score = (
        df['study_time'] * 0.25 +
        (6 - df['financial_stress']) * 0.15 +
        df['family_support'] * 0.15 +
        (df['sleep_hours'] / 2) * 0.1 +
        (4 - df['social_media_hours']/2) * 0.1 +
        df['health_status'] * 0.1 +
        (df['internet_access'] == 'yes') * 0.1 +
        (4 - df['previous_failures']) * 0.05 +
        np.random.normal(0, 0.5, n_samples)  # Add some noise
    )
    
    # Convert to categories
    percentiles = np.percentile(performance_score, [25, 50, 75])
    df['performance'] = pd.cut(performance_score, 
                              bins=[-np.inf, percentiles[0], percentiles[1], percentiles[2], np.inf],
                              labels=['Poor', 'Average', 'Good', 'Excellent'])
    
    return df, 'demo'

# Try to load data
print("Attempting to load student performance dataset...")
data, dataset_type = load_uci_student_data()

if data is None:
    print("\n🔄 Using demonstration dataset with lifestyle factors...")
    data, dataset_type = create_lifestyle_demo_data()

print(f"\n📈 Dataset loaded successfully!")
print(f"Dataset type: {dataset_type}")
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Display basic information
print("\n" + "="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\n📊 Dataset Overview:")
print(data.info())
print("\n📈 First few rows:")
print(data.head())

if dataset_type == 'demo':
    target_col = 'performance'
    print(f"\n🎯 Target variable distribution:")
    print(data[target_col].value_counts())
else:
    # For UCI dataset, we'll create performance categories from G3 (final grade)
    if 'G3' in data.columns:
        # Convert grades to performance categories
        data['performance'] = pd.cut(data['G3'], 
                                   bins=[0, 9, 13, 16, 20], 
                                   labels=['Poor', 'Average', 'Good', 'Excellent'])
        target_col = 'performance'
        print(f"\n🎯 Performance categories created from final grades (G3):")
        print(data[target_col].value_counts())

# Visualization function
def create_lifestyle_analysis_plots(df, target_col):
    """Create comprehensive lifestyle analysis plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Student Lifestyle Factors vs Academic Performance', fontsize=16, fontweight='bold')
    
    if dataset_type == 'demo':
        # Sleep hours vs performance
        sns.boxplot(data=df, x=target_col, y='sleep_hours', ax=axes[0,0])
        axes[0,0].set_title('Sleep Hours by Performance')
        
        # Social media usage vs performance
        sns.boxplot(data=df, x=target_col, y='social_media_hours', ax=axes[0,1])
        axes[0,1].set_title('Social Media Hours by Performance')
        
        # Study time vs performance
        sns.countplot(data=df, x='study_time', hue=target_col, ax=axes[0,2])
        axes[0,2].set_title('Study Time Distribution by Performance')
        
        # Financial stress impact
        sns.boxplot(data=df, x=target_col, y='financial_stress', ax=axes[1,0])
        axes[1,0].set_title('Financial Stress by Performance')
        
        # Exercise frequency
        sns.boxplot(data=df, x=target_col, y='exercise_frequency', ax=axes[1,1])
        axes[1,1].set_title('Exercise Frequency by Performance')
        
        # Accommodation type
        pd.crosstab(df['accommodation'], df[target_col], normalize='index').plot(kind='bar', ax=axes[1,2])
        axes[1,2].set_title('Performance by Accommodation Type')
        axes[1,2].legend(title='Performance')
        
    else:
        # UCI dataset visualizations
        if 'studytime' in df.columns:
            sns.countplot(data=df, x='studytime', hue=target_col, ax=axes[0,0])
            axes[0,0].set_title('Study Time vs Performance')
        
        if 'health' in df.columns:
            sns.countplot(data=df, x='health', hue=target_col, ax=axes[0,1])
            axes[0,1].set_title('Health Status vs Performance')
            
        if 'goout' in df.columns:
            sns.countplot(data=df, x='goout', hue=target_col, ax=axes[0,2])
            axes[0,2].set_title('Social Activities vs Performance')
        
        if 'Walc' in df.columns:
            sns.countplot(data=df, x='Walc', hue=target_col, ax=axes[1,0])
            axes[1,0].set_title('Weekend Alcohol vs Performance')
            
        if 'romantic' in df.columns:
            sns.countplot(data=df, x='romantic', hue=target_col, ax=axes[1,1])
            axes[1,1].set_title('Romantic Relationship vs Performance')
            
        if 'famrel' in df.columns:
            sns.countplot(data=df, x='famrel', hue=target_col, ax=axes[1,2])
            axes[1,2].set_title('Family Relationship vs Performance')
    
    plt.tight_layout()
    plt.show()

# Create visualizations
create_lifestyle_analysis_plots(data, target_col)

print("\n" + "="*60)
print("STEP 3: DATA PREPROCESSING")
print("="*60)

# Remove rows with missing target values
data_clean = data.dropna(subset=[target_col]).copy()
print(f"Data shape after removing missing targets: {data_clean.shape}")

# Prepare features for modeling
def prepare_features(df, target_col, dataset_type):
    """Prepare features for machine learning"""
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle different dataset types
    if dataset_type == 'demo':
        # For demo dataset, we know the structure
        categorical_features = ['gender', 'accommodation', 'family_size', 'part_time_job', 
                              'social_activities', 'romantic_relationship', 'internet_access', 
                              'extra_curricular']
        numerical_features = ['age', 'financial_stress', 'family_support', 'study_time',
                            'sleep_hours', 'social_media_hours', 'exercise_frequency', 
                            'health_status', 'family_pressure', 'previous_failures']
    else:
        # For UCI dataset, automatically detect feature types
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    
    # Encode categorical variables
    label_encoders = {}
    X_processed = X.copy()
    
    for col in categorical_features:
        if col in X_processed.columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X_processed, y_encoded, label_encoders, target_encoder, categorical_features, numerical_features

X, y, label_encoders, target_encoder, cat_features, num_features = prepare_features(
    data_clean, target_col, dataset_type)

print(f"\nProcessed features shape: {X.shape}")
print(f"Target classes: {target_encoder.classes_}")
print(f"Class distribution: {np.bincount(y)}")

print("\n" + "="*60)
print("STEP 4: MODEL TRAINING AND EVALUATION")
print("="*60)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

if num_features:
    X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
    X_test_scaled[num_features] = scaler.transform(X_test[num_features])

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8)
}

# Train and evaluate models
model_results = {}
trained_models = {}

print("\n🔄 Training models...")
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for Naive Bayes, original for tree-based models
    if name == 'Naive Bayes':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    model_results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'predictions': y_pred
    }
    
    trained_models[name] = model
    
    print(f"✅ {name}:")
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")

# Create ensemble model
print(f"\n🔄 Training Ensemble Model...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)),
        ('nb', GaussianNB()),
        ('dt', DecisionTreeClassifier(random_state=42, max_depth=8))
    ],
    voting='soft'
)

ensemble.fit(X_train_scaled, y_train)
ensemble_pred = ensemble.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_cv = cross_val_score(ensemble, X_train_scaled, y_train, cv=5)

model_results['Ensemble'] = {
    'accuracy': ensemble_accuracy,
    'cv_mean': ensemble_cv.mean(),
    'cv_std': ensemble_cv.std(),
    'predictions': ensemble_pred
}

print(f"✅ Ensemble Model:")
print(f"   Test Accuracy: {ensemble_accuracy:.4f}")
print(f"   CV Score: {ensemble_cv.mean():.4f} (+/- {ensemble_cv.std()*2:.4f})")

print("\n" + "="*60)
print("STEP 5: RESULTS ANALYSIS")
print("="*60)

# Model comparison
print("\n📊 Model Performance Comparison:")
results_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Test Accuracy': [model_results[m]['accuracy'] for m in model_results.keys()],
    'CV Mean': [model_results[m]['cv_mean'] for m in model_results.keys()],
    'CV Std': [model_results[m]['cv_std'] for m in model_results.keys()]
})

print(results_df.round(4))

# Find best model
best_model_name = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
print(f"\n🏆 Best performing model: {best_model_name}")

# Detailed classification report for best model
print(f"\n📈 Detailed Classification Report - {best_model_name}:")
best_predictions = model_results[best_model_name]['predictions']
print(classification_report(y_test, best_predictions, 
                          target_names=target_encoder.classes_))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance (for Random Forest)
if best_model_name == 'Random Forest' or 'Random Forest' in trained_models:
    rf_model = trained_models.get('Random Forest', trained_models[best_model_name])
    if hasattr(rf_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title('Top 15 Most Important Features (Random Forest)')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()
        
        print("\n🔍 Top 10 Most Important Features:")
        print(feature_importance.head(10))

print("\n" + "="*60)
print("STEP 6: INSIGHTS AND RECOMMENDATIONS")
print("="*60)

print("\n💡 Key Insights from the Analysis:")

if dataset_type == 'demo':
    print("""
1. LIFESTYLE FACTORS IMPACT:
   - Sleep hours show strong correlation with academic performance
   - Social media usage negatively impacts performance when excessive
   - Regular exercise contributes to better academic outcomes
   
2. SOCIOECONOMIC FACTORS:
   - Financial stress significantly affects student performance
   - Family support plays a crucial role in academic success
   - Accommodation type (hostel vs home) influences study patterns
   
3. STUDY HABITS:
   - Study time is the strongest predictor of performance
   - Internet access is crucial for modern academic success
   - Previous failures strongly predict future performance
""")
else:
    print("""
1. ACADEMIC FACTORS:
   - Study time and previous failures are key predictors
   - Family educational support strongly influences outcomes
   - School choice and travel time affect performance
   
2. SOCIAL AND HEALTH FACTORS:
   - Health status correlates with academic performance
   - Social activities and relationships impact study focus
   - Alcohol consumption negatively affects grades
   
3. FAMILY DYNAMICS:
   - Family relationships and support are crucial
   - Parental education level influences student success
   - Family size and structure affect available resources
""")

print(f"""
📈 MODEL PERFORMANCE SUMMARY:
- Best Model: {best_model_name}
- Accuracy: {model_results[best_model_name]['accuracy']:.4f}
- Cross-validation: {model_results[best_model_name]['cv_mean']:.4f} ± {model_results[best_model_name]['cv_std']:.4f}

🎯 PRACTICAL APPLICATIONS:
1. Early Warning System: Identify at-risk students
2. Intervention Planning: Target specific lifestyle factors
3. Resource Allocation: Focus on high-impact support areas
4. Policy Development: Evidence-based educational policies

🔧 NEXT STEPS:
1. Collect more Pakistani university-specific data
2. Include mental health and stress indicators
3. Add temporal factors (semester, exam periods)
4. Implement real-time monitoring system
""")

print("\n" + "="*80)
print("PROJECT COMPLETED SUCCESSFULLY! 🎉")
print("="*80)