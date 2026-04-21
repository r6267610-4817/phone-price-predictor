"""
Smartphone Price Prediction & Condition Analysis System
Combining Machine Learning, LLM Integration, and Image Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from io import BytesIO
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Image processing
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Smartphone Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 2rem;
        font-weight: bold;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        transition: 0.3s;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <h1>📱 Smartphone Price Prediction & Condition Analysis</h1>
    <p>AI-Powered Price Estimation | Condition Assessment | Smart Recommendations</p>
</div>
""", unsafe_allow_html=True)

# ==================== Data loading and preprocessing ====================
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the Carousell dataset"""
    df = pd.read_csv('carousell_regression_numeric_y_price_add_storage.csv')
    
    # Clean price column - remove extreme outliers
    df = df[(df['Price'] > 0) & (df['Price'] < 100000)]
    df = df[df['Original_Price_HKD'] > 0]
    
    # Remove extreme outliers using IQR method (as in notebook)
    Q1 = df['Price'].quantile(0.01)
    Q3 = df['Price'].quantile(0.99)
    df = df[(df['Price'] >= Q1) & (df['Price'] <= Q3)]
    
    # Handle missing values
    df = df.fillna(df.median())
    
    # Create derived features
    df['Price_Discount_Pct'] = ((df['Original_Price_HKD'] - df['Price']) / df['Original_Price_HKD']) * 100
    df['Price_Discount_Pct'] = df['Price_Discount_Pct'].clip(0, 100)
    
    df['Age_Months'] = df['Months_Since_Release']
    df['Price_per_GB'] = df['Price'] / df['Storage_GB'].replace(0, 1)
    df['Price_per_GB'] = df['Price_per_GB'].clip(0, 5000)
    
    df['Value_Retention'] = (df['Price'] / df['Original_Price_HKD']) * 100
    df['Value_Retention'] = df['Value_Retention'].clip(0, 150)
    
    df['Normalized_Condition'] = df['Condition_Percentage'] / 100
    df['Battery_Health_Norm'] = df['Battery_Health_Percent'] / 100
    df['Seller_Reputation_Norm'] = np.log1p(df['seller_reviews_count'])
    
    # Brand encoding
    brand_cols = ['Brand_Apple', 'Brand_Huawei', 'Brand_Samsung', 'Brand_Sony', 'Brand_Xiaomi']
    df['Brand'] = df[brand_cols].idxmax(axis=1).str.replace('Brand_', '')
    
    return df

@st.cache_data
def prepare_features(df):
    """Prepare features for machine learning - using top features from consensus ranking"""
    # Selected features with Average_Rank < 10 from consensus feature ranking
    selected_features = [
        'Months_Since_Release',
        'Condition_Score', 
        'Storage_GB',
        'Product_Origin_Unknown',
        'Is_Premium_Tier',
        'Post_Recency_Log',
        'Days_Since_Posted',
        'Original_Price_HKD',
        'Brand_Apple'
    ]
    
    # Add binary features for completeness
    binary_cols = ['Is_Working', 'Is_MTR_Trade', 'Has_Warranty', 'Has_Box',
                   'Has_Receipt', 'Has_Accessories', 'Is_Firm_Price',
                   'Is_Urgent_Sale', 'Has_Repair_History', 'Has_Crack_Or_Line',
                   'Has_Dual_SIM', 'Has_Brand_Keyword_Mismatch']
    
    # Add brand dummies for other brands (not in selected features)
    brand_dummies = pd.get_dummies(df['Brand'], prefix='Brand')
    # Keep only non-Apple brand dummies since Brand_Apple is already selected
    other_brand_cols = [col for col in brand_dummies.columns if col != 'Brand_Apple']
    
    X = df[selected_features].copy()
    X = pd.concat([X, df[binary_cols], brand_dummies[other_brand_cols]], axis=1)
    
    # Handle infinity and NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Target: log(Price + 1) as per notebook analysis
    y_log = np.log1p(df['Price'])
    
    return X, y_log, selected_features

# Load data
with st.spinner("Loading and processing data..."):
    df = load_and_preprocess_data()
    X, y_log, selected_features = prepare_features(df)

st.success(f"✅ Data loaded successfully! {len(df)} listings analyzed")

# ==================== Sidebar Filters ====================
st.sidebar.header("🔍 Data Filters")

with st.sidebar.expander("Filter Listings", expanded=False):
    brand_filter = st.multiselect(
        "Brand",
        options=df['Brand'].unique(),
        default=df['Brand'].unique()[:3]
    )
    
    price_range = st.slider(
        "Price Range (HKD)",
        min_value=int(df['Price'].min()),
        max_value=int(df['Price'].max()),
        value=(int(df['Price'].quantile(0.1)), int(df['Price'].quantile(0.9)))
    )
    
    storage_filter = st.multiselect(
        "Storage (GB)",
        options=sorted(df['Storage_GB'].unique()),
        default=[]
    )

# Apply filters
filtered_df = df.copy()
if brand_filter:
    filtered_df = filtered_df[filtered_df['Brand'].isin(brand_filter)]
if storage_filter:
    filtered_df = filtered_df[filtered_df['Storage_GB'].isin(storage_filter)]
filtered_df = filtered_df[(filtered_df['Price'] >= price_range[0]) & 
                          (filtered_df['Price'] <= price_range[1])]

# ==================== EDA Dashboard ====================
st.header("📊 Market Analysis Dashboard")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Listings", f"{len(filtered_df):,}", 
              delta=f"{len(filtered_df)/len(df)*100:.0f}% of total")
with col2:
    st.metric("Avg Price", f"HK${filtered_df['Price'].mean():,.0f}",
              delta=f"${filtered_df['Price'].mean() - df['Price'].mean():.0f}")
with col3:
    st.metric("Avg Discount", f"{filtered_df['Price_Discount_Pct'].mean():.1f}%")
with col4:
    st.metric("Avg Condition", f"{filtered_df['Condition_Percentage'].mean():.1f}%")

# Charts
tab1, tab2, tab3 = st.tabs(["📈 Price Distribution", "🏷️ Brand Analysis", "📉 Price Factors"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fig_price = px.histogram(
            filtered_df, x='Price', nbins=50,
            title='Price Distribution',
            color_discrete_sequence=['#667eea']
        )
        fig_price.update_layout(showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        fig_box = px.box(
            filtered_df, x='Brand', y='Price',
            title='Price by Brand',
            color='Brand'
        )
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    brand_stats = filtered_df.groupby('Brand').agg({
        'Price': ['mean', 'count', 'median'],
        'Condition_Percentage': 'mean'
    }).round(2)
    brand_stats.columns = ['Avg Price', 'Count', 'Median Price', 'Avg Condition']
    brand_stats = brand_stats.sort_values('Avg Price', ascending=False)
    st.dataframe(brand_stats, use_container_width=True)
    
    fig_brand = px.bar(
        brand_stats.reset_index(),
        x='Brand', y='Avg Price',
        title='Average Price by Brand',
        color='Avg Price',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_brand, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig_storage = px.box(
            filtered_df, x='Storage_GB', y='Price',
            title='Price vs Storage',
            color='Storage_GB'
        )
        st.plotly_chart(fig_storage, use_container_width=True)
    
    with col2:
        # Removed trendline='ols' to avoid statsmodels dependency
        fig_condition = px.scatter(
            filtered_df, x='Condition_Percentage', y='Price',
            title='Price vs Condition',
            color='Brand'
        )
        st.plotly_chart(fig_condition, use_container_width=True)

# ==================== Machine Learning Model ====================
st.header("🤖 Price Prediction Model")

@st.cache_resource
def train_models():
    """Train optimized models based on notebook analysis"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    # Store feature names before scaling
    feature_names_list = X.columns.tolist()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Gradient Boosting - Best model from notebook (R2_Log: 0.4383)
    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)
    
    # Random Forest - Second best model (R2_Log: 0.4256)
    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    # Make predictions
    models = {
        'Gradient Boosting': gb,
        'Random Forest': rf,
        'Ridge Regression': ridge
    }
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        # Convert from log scale back to original price for error metrics
        y_pred_original = np.expm1(y_pred)
        y_test_original = np.expm1(y_test)
        
        results[name] = {
            'model': model,
            'rmse': np.sqrt(mean_squared_error(y_test_original, y_pred_original)),
            'mae': mean_absolute_error(y_test_original, y_pred_original),
            'r2_log': r2_score(y_test, y_pred),
            'predictions': y_pred,
            'actual': y_test.values
        }
    
    return results, scaler, feature_names_list

with st.spinner("Training AI models..."):
    model_results, scaler, feature_names = train_models()

# Model comparison
st.subheader("📊 Model Performance Comparison")

col1, col2, col3 = st.columns(3)
for i, (name, results) in enumerate(model_results.items()):
    with [col1, col2, col3][i]:
        badge = "🥇 Best" if name == "Gradient Boosting" else "🥈" if name == "Random Forest" else "📊"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{badge} {name}</h3>
            <p>RMSE: <b>HK${results['rmse']:.0f}</b></p>
            <p>MAE: <b>HK${results['mae']:.0f}</b></p>
            <p>R² (log scale): <b>{results['r2_log']:.3f}</b></p>
        </div>
        """, unsafe_allow_html=True)

# Feature importance for Gradient Boosting (best model)
best_model_name = "Gradient Boosting"
best_model = model_results[best_model_name]['model']

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    fig_importance = px.bar(
        feature_importance, x='Importance', y='Feature',
        orientation='h', title=f'Top 15 Features - {best_model_name}',
        color='Importance', color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.caption("📊 Features selected via consensus ranking (Lasso + Random Forest + Gradient Boosting)")

# ==================== Interactive Price Prediction ====================
st.header("🎯 Interactive Price Predictor")

st.markdown("Adjust the phone specifications below to get an AI-powered price estimate")

col1, col2, col3 = st.columns(3)

with col1:
    brand_input = st.selectbox("Brand", options=df['Brand'].unique())
    storage_input = st.selectbox("Storage (GB)", options=sorted(df['Storage_GB'].unique()))
    condition_input = st.slider("Condition (%)", 0, 100, 90)

with col2:
    months_input = st.slider("Months Since Release", 0, 120, 24)
    battery_input = st.slider("Battery Health (%)", 50, 100, 85)
    warranty_input = st.selectbox("Has Warranty?", ["Yes", "No"])

with col3:
    box_input = st.selectbox("Has Original Box?", ["Yes", "No"])
    accessories_input = st.selectbox("Has Accessories?", ["Yes", "No"])
    repair_input = st.selectbox("Has Repair History?", ["No", "Yes"])

# Prepare input features based on selected features
def prepare_prediction_input():
    input_dict = {}
    
    # Get median values from training data for reference
    original_price_median = df['Original_Price_HKD'].median()
    post_recency_median = df['Post_Recency_Log'].median()
    days_since_posted_median = df['Days_Since_Posted'].median()
    product_origin_unknown_default = df['Product_Origin_Unknown'].mode()[0] if len(df['Product_Origin_Unknown'].mode()) > 0 else 0
    is_premium_tier_default = df['Is_Premium_Tier'].mode()[0] if len(df['Is_Premium_Tier'].mode()) > 0 else 0
    
    for feat in feature_names:
        if feat == 'Storage_GB':
            input_dict[feat] = storage_input
        elif feat == 'Months_Since_Release':
            input_dict[feat] = months_input
        elif feat == 'Condition_Score':
            # Map condition percentage to Condition_Score (similar scale 0-100)
            input_dict[feat] = condition_input
        elif feat == 'Battery_Health_Percent':
            input_dict[feat] = battery_input
        elif feat == 'Original_Price_HKD':
            input_dict[feat] = original_price_median
        elif feat == 'Post_Recency_Log':
            input_dict[feat] = post_recency_median
        elif feat == 'Days_Since_Posted':
            input_dict[feat] = days_since_posted_median
        elif feat == 'Product_Origin_Unknown':
            input_dict[feat] = product_origin_unknown_default
        elif feat == 'Is_Premium_Tier':
            input_dict[feat] = is_premium_tier_default
        elif feat.startswith('Brand_'):
            input_dict[feat] = 1 if feat == f'Brand_{brand_input}' else 0
        elif feat in ['Has_Warranty', 'Has_Box', 'Has_Accessories', 
                      'Has_Repair_History', 'Is_Working', 'Is_MTR_Trade',
                      'Has_Receipt', 'Is_Firm_Price', 'Is_Urgent_Sale',
                      'Has_Crack_Or_Line', 'Has_Dual_SIM',
                      'Has_Brand_Keyword_Mismatch']:
            input_dict[feat] = 0
        else:
            input_dict[feat] = X[feat].median() if feat in X.columns else 0
    
    # Override specific binary features
    input_dict['Has_Warranty'] = 1 if warranty_input == "Yes" else 0
    input_dict['Has_Box'] = 1 if box_input == "Yes" else 0
    input_dict['Has_Accessories'] = 1 if accessories_input == "Yes" else 0
    input_dict['Has_Repair_History'] = 1 if repair_input == "Yes" else 0
    
    input_df = pd.DataFrame([input_dict])[feature_names]
    input_df = input_df.fillna(0)
    
    return input_df

input_features = prepare_prediction_input()
input_scaled = scaler.transform(input_features)

# Get predictions from all models
predictions = {}
for name, results in model_results.items():
    pred_log = results['model'].predict(input_scaled)[0]
    pred_original = np.expm1(pred_log)
    predictions[name] = pred_original

# Display predictions
st.subheader("📈 AI Price Predictions")

pred_cols = st.columns(len(predictions))
for i, (name, pred) in enumerate(predictions.items()):
    with pred_cols[i]:
        badge = "🏆" if name == "Gradient Boosting" else "📊"
        st.markdown(f"""
        <div class="prediction-card">
            <h3>{badge} {name}</h3>
            <h2 style="color: #667eea;">HK${pred:,.0f}</h2>
            <p>R² (log scale): {(model_results[name]['r2_log'] * 100):.0f}%</p>
        </div>
        """, unsafe_allow_html=True)

# Ensemble prediction (weighted by R² scores)
weights = {name: results['r2_log'] for name, results in model_results.items()}
total_weight = sum(weights.values())
ensemble_pred = sum(predictions[name] * weights[name] / total_weight for name in predictions.keys())

st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 1rem; text-align: center; margin-top: 1rem;">
    <h3 style="color: white;">🎯 Weighted Ensemble Prediction</h3>
    <h1 style="color: white; font-size: 3rem;">HK${ensemble_pred:,.0f}</h1>
    <p style="color: white;">Based on {len(predictions)} AI models (weighted by R²)</p>
</div>
""", unsafe_allow_html=True)

# ==================== AI LLM Integration ====================
st.header("🤖 AI Assistant - Smart Recommendations")

st.markdown("Get personalized buying/selling recommendations from our AI assistant")

def get_llm_recommendation():
    context = f"""
    Phone Specifications:
    - Brand: {brand_input}
    - Storage: {storage_input} GB
    - Condition: {condition_input}%
    - Months since release: {months_input}
    - Battery health: {battery_input}%
    - Has warranty: {warranty_input}
    - Has original box: {box_input}
    
    AI Price Predictions:
    - Gradient Boosting: HK${predictions.get('Gradient Boosting', 0):,.0f}
    - Random Forest: HK${predictions.get('Random Forest', 0):,.0f}
    - Ridge Regression: HK${predictions.get('Ridge Regression', 0):,.0f}
    - Ensemble: HK${ensemble_pred:,.0f}
    
    Market Context:
    - Average price for {brand_input}: HK${filtered_df[filtered_df['Brand']==brand_input]['Price'].mean():,.0f}
    - Typical discount rate: {filtered_df['Price_Discount_Pct'].mean():.1f}%
    """
    return context

recommendation_context = get_llm_recommendation()

with st.expander("📝 AI Analysis & Recommendations", expanded=True):
    st.markdown("### 📊 Market Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        brand_avg = filtered_df[filtered_df['Brand']==brand_input]['Price'].mean()
        if ensemble_pred > brand_avg * 1.1:
            st.info(f"📈 This device is priced **{((ensemble_pred/brand_avg)-1)*100:.0f}% above** the average {brand_input} price")
        elif ensemble_pred < brand_avg * 0.9:
            st.success(f"📉 This device is priced **{(1 - ensemble_pred/brand_avg)*100:.0f}% below** the average {brand_input} price")
        else:
            st.info(f"📊 This device is priced **around market average** for {brand_input}")
    
    with col2:
        condition_impact = filtered_df.groupby('Brand')['Condition_Percentage'].mean()
        if condition_input > condition_impact.get(brand_input, 80):
            st.success(f"✨ Above-average condition (+{condition_input - condition_impact.get(brand_input, 80):.0f}%) → Higher value")
        elif condition_input < condition_impact.get(brand_input, 80):
            st.warning(f"⚠️ Below-average condition ({condition_input - condition_impact.get(brand_input, 80):.0f}%) → Lower value")
        else:
            st.info("✅ Average condition for this brand")
    
    st.markdown("### 💡 Smart Recommendations")
    
    recommendations = []
    
    if condition_input < 70:
        recommendations.append("🔧 Consider professional cleaning/repair before selling to increase value")
    if not warranty_input == "Yes" and months_input < 12:
        recommendations.append("📋 Device is relatively new but lacks warranty - highlight its excellent condition")
    if ensemble_pred > filtered_df['Price'].quantile(0.75):
        recommendations.append("💰 Premium pricing range - target collectors or brand enthusiasts")
    elif ensemble_pred < filtered_df['Price'].quantile(0.25):
        recommendations.append("🎯 Great value deal - quick sale expected if priced competitively")
    
    if not recommendations:
        recommendations.append("✅ Fairly priced device with good market positioning")
    
    for rec in recommendations:
        st.write(rec)
    
    st.markdown("### 🧠 AI Deep Analysis")
    
    if ensemble_pred > 8000:
        analysis = "This is a premium-tier device. The high price reflects strong brand value and likely good condition. Consider highlighting unique features and providing detailed photos to justify the price point."
    elif ensemble_pred > 4000:
        analysis = "Mid-range pricing detected. This segment is highly competitive. Emphasize battery health, included accessories, and any remaining warranty to stand out from similar listings."
    else:
        analysis = "Budget-friendly pricing. Focus on value proposition - good condition, functional device at an accessible price point. Quick sale expected in this segment."
    
    st.info(f"💭 {analysis}")
    
    st.markdown("### 📝 Selling Tips")
    
    tips = [
        f"📸 Include {max(5, int(filtered_df['image_count'].mean()))}+ high-quality photos",
        "📝 Write detailed description highlighting unique features",
        "🏷️ Price competitively - consider 5-10% below similar listings for quick sale",
        "✅ Mention all included accessories and original packaging",
        "⭐ Respond to inquiries quickly - top sellers have faster response times"
    ]
    
    for tip in tips:
        st.write(tip)

# ==================== Phone Condition Image Analysis ====================
st.header("📸 Phone Condition Analysis (AI Vision Detection)")

st.markdown("Upload a phone photo for AI-powered condition assessment (screen condition, body wear, overall score)")

uploaded_file = st.file_uploader(
    "Choose a phone image...",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Upload a clear photo of the phone (good lighting, front-facing angle recommended)"
)

def extract_image_features(image):
    """Extract image features for analysis"""
    img_array = np.array(image.convert('L'))
    
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    sharpness = np.std(np.diff(img_array, axis=0)) + np.std(np.diff(img_array, axis=1))
    
    from scipy import ndimage
    
    edges = ndimage.sobel(img_array)
    edge_density = np.mean(np.abs(edges) > 30)
    
    uniformity = 1 - np.std(img_array) / np.mean(img_array) if np.mean(img_array) > 0 else 0
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'sharpness': sharpness,
        'edge_density': edge_density,
        'uniformity': uniformity
    }

def analyze_phone_condition(image, features):
    """Analyze phone condition based on image features"""
    
    if features['edge_density'] > 0.15:
        screen_condition = "⚠️ Potential scratches/cracks detected"
        screen_score = 60
    elif features['edge_density'] > 0.08:
        screen_condition = "📱 Minor signs of use"
        screen_score = 75
    else:
        screen_condition = "✨ Screen in excellent condition"
        screen_score = 90
    
    if features['uniformity'] < 0.6:
        body_condition = "🔄 Visible wear on body"
        body_score = 65
    elif features['uniformity'] < 0.75:
        body_condition = "👍 Normal signs of use"
        body_score = 80
    else:
        body_condition = "🌟 Body looks like new"
        body_score = 95
    
    if features['sharpness'] < 50:
        image_quality = "📷 Image quality is low, consider retaking the photo"
    else:
        image_quality = "✅ Image is clear, accurate assessment possible"
    
    overall_score = int((screen_score + body_score) / 2)
    
    return {
        'screen_condition': screen_condition,
        'screen_score': screen_score,
        'body_condition': body_condition,
        'body_score': body_score,
        'image_quality': image_quality,
        'overall_score': overall_score
    }

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    max_size = 400
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Phone Image", use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 AI Vision Analysis Results")
        
        with st.spinner("Analyzing image..."):
            features = extract_image_features(image)
            analysis = analyze_phone_condition(image, features)
        
        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            <p><strong>📱 Screen Condition:</strong> {analysis['screen_condition']}</p>
            <p><strong>🔧 Body Condition:</strong> {analysis['body_condition']}</p>
            <p><strong>{analysis['image_quality']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(analysis['overall_score'] / 100)
        st.metric("📊 Overall Condition Score", f"{analysis['overall_score']}/100")
        
        if analysis['overall_score'] >= 85:
            st.success("✅ Phone is in excellent condition! Can be listed as 'Like New' or 'Minor signs of use'")
            price_adjustment = "+5%"
        elif analysis['overall_score'] >= 70:
            st.info("👍 Phone is in good condition with normal wear, price at market average")
            price_adjustment = "0%"
        else:
            st.warning("⚠️ Phone shows significant wear, consider lowering price for faster sale")
            price_adjustment = "-10%"
        
        st.caption(f"💡 Based on image analysis, suggested price adjustment: {price_adjustment}")
        
        with st.expander("📊 Detailed Image Metrics"):
            st.json({
                "Brightness": f"{features['brightness']:.1f}",
                "Contrast": f"{features['contrast']:.1f}",
                "Sharpness": f"{features['sharpness']:.1f}",
                "Edge Density": f"{features['edge_density']:.3f}",
                "Uniformity": f"{features['uniformity']:.3f}"
            })
else:
    st.info("👆 Please upload a phone photo for AI vision analysis")
    st.markdown("""
    ### 📸 Photo Tips:
    - Take photos in well-lit environment
    - Include front, back, and edge photos of the phone
    - Take close-up photos of any scratches or damage
    - Avoid glare and shadows
    """)

# ==================== Market Insights ====================
st.header("📈 Market Insights & Trends")

col1, col2 = st.columns(2)

with col1:
    storage_trend = filtered_df.groupby('Storage_GB')['Price'].agg(['mean', 'count']).reset_index()
    storage_trend = storage_trend[storage_trend['count'] > 5]
    
    fig_trend = px.line(
        storage_trend, x='Storage_GB', y='mean',
        title='Price Trend by Storage Capacity',
        markers=True,
        labels={'mean': 'Average Price (HKD)', 'Storage_GB': 'Storage (GB)'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col2:
    brand_condition = filtered_df.groupby(['Brand', pd.cut(filtered_df['Condition_Percentage'], 
                                                           bins=[0, 70, 85, 100], 
                                                           labels=['Poor', 'Good', 'Excellent'])]).size().unstack()
    
    fig_condition_stack = px.bar(
        brand_condition.reset_index(),
        x='Brand', y=['Poor', 'Good', 'Excellent'],
        title='Condition Distribution by Brand',
        barmode='stack',
        color_discrete_sequence=['#ff6b6b', '#ffd93d', '#6bcb77']
    )
    st.plotly_chart(fig_condition_stack, use_container_width=True)

st.subheader("📉 Value Depreciation Analysis")

depreciation = filtered_df.groupby('Months_Since_Release')['Value_Retention'].mean().reset_index()
depreciation = depreciation[depreciation['Months_Since_Release'] <= 60]

fig_depreciation = px.scatter(
    depreciation, x='Months_Since_Release', y='Value_Retention',
    title='Value Retention Over Time',
    trendline='lowess',
    labels={'Months_Since_Release': 'Months Since Release', 
            'Value_Retention': 'Value Retention (%)'}
)
fig_depreciation.update_traces(marker=dict(size=8, color='#667eea'))
st.plotly_chart(fig_depreciation, use_container_width=True)

# ==================== Download Predictions ====================
st.sidebar.markdown("---")
st.sidebar.header("💾 Export Data")

if st.sidebar.button("📥 Download Predictions"):
    export_df = pd.DataFrame({
        'Model': list(predictions.keys()),
        'Predicted_Price_HKD': list(predictions.values())
    })
    export_df.loc[len(export_df)] = ['Ensemble', ensemble_pred]
    
    csv = export_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📊 Download as CSV",
        data=csv,
        file_name="phone_price_predictions.csv",
        mime="text/csv"
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📱 System Info
- **Total Listings:** {:,}
- **Brands:** {}
- **Avg Price:** HK${:,.0f}
- **Best Model:** {}
- **R² (log scale):** {:.3f}
""".format(
    len(df),
    df['Brand'].nunique(),
    df['Price'].mean(),
    best_model_name,
    model_results[best_model_name]['r2_log']
))

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🚀 Features
- 🤖 Multi-Model AI Prediction
- 📸 Image-based Condition Analysis
- 🧠 AI-Powered Recommendations
- 📊 Real-time Market Analytics
- 💾 Export Predictions
""")

if __name__ == "__main__":
    pass
