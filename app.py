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
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
import joblib
import pickle
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
    """Load and preprocess the Carousell dataset - MATCHING IPYNB EXACTLY"""
    df = pd.read_csv('carousell_regression_numeric_y_price_add_storage.csv')
    
    # Clean price column
    df = df[(df['Price'] > 0) & (df['Original_Price_HKD'] > 0)]
    
    # Use EXACT same IQR outlier removal as in ipynb
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]
    
    # Handle missing values
    df = df.fillna(df.median())
    
    # Create derived features (for EDA only, not for model)
    df['Price_Discount_Pct'] = ((df['Original_Price_HKD'] - df['Price']) / df['Original_Price_HKD']) * 100
    df['Price_Discount_Pct'] = df['Price_Discount_Pct'].clip(0, 100)
    df['Value_Retention'] = (df['Price'] / df['Original_Price_HKD']) * 100
    df['Value_Retention'] = df['Value_Retention'].clip(0, 150)
    
    # Brand encoding for EDA
    brand_cols = ['Brand_Apple', 'Brand_Huawei', 'Brand_Samsung', 'Brand_Sony', 'Brand_Xiaomi']
    df['Brand'] = df[brand_cols].idxmax(axis=1).str.replace('Brand_', '')
    
    return df

@st.cache_data
def prepare_features(df):
    """Prepare features - EXACTLY matching ipynb consensus feature selection"""
    # Selected features with Average_Rank < 10 from consensus feature ranking (ipynb)
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
    
    X = df[selected_features].copy()
    
    # Handle infinity and NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X, selected_features

# Load saved models
@st.cache_resource
def load_saved_models():
    """Load pre-trained models from ipynb"""
    try:
        model = joblib.load('gradient_boosting_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('model_metrics.pkl', 'rb') as f:
            model_metrics = pickle.load(f)
        
        return model, scaler, feature_names, model_metrics
    except FileNotFoundError as e:
        st.error(f"模型文件未找到: {e}")
        st.info("请先在 ipynb 中运行模型保存代码，生成以下文件：\n"
                "- gradient_boosting_model.pkl\n"
                "- scaler.pkl\n"
                "- feature_names.pkl\n"
                "- model_metrics.pkl")
        st.stop()
        return None, None, None, None

# Load data
with st.spinner("Loading and processing data..."):
    df = load_and_preprocess_data()
    X, selected_features = prepare_features(df)

st.success(f"✅ Data loaded successfully! {len(df):,} listings analyzed")

# Load saved models
with st.spinner("Loading pre-trained AI models..."):
    best_model, scaler, feature_names, model_metrics = load_saved_models()

# ==================== Sidebar Filters ====================
st.sidebar.header("🔍 Data Filters")

with st.sidebar.expander("Filter Listings", expanded=False):
    brand_filter = st.multiselect(
        "Brand",
        options=df['Brand'].unique(),
        default=df['Brand'].unique()[:3] if len(df['Brand'].unique()) >= 3 else df['Brand'].unique()
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
if len(filtered_df) > 0:
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
    if len(filtered_df) > 0:
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
    if len(filtered_df) > 0:
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
    if len(filtered_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig_storage = px.box(
                filtered_df, x='Storage_GB', y='Price',
                title='Price vs Storage',
                color='Storage_GB'
            )
            st.plotly_chart(fig_storage, use_container_width=True)
        
        with col2:
            fig_condition = px.scatter(
                filtered_df, x='Condition_Percentage', y='Price',
                title='Price vs Condition',
                color='Brand',
                trendline='ols'
            )
            st.plotly_chart(fig_condition, use_container_width=True)

# ==================== Model Performance Display ====================
st.header("🤖 Price Prediction Model")

st.subheader("📊 Model Performance (from Notebook)")

# Display metrics from saved model
gb_metrics = model_metrics.get('Gradient Boosting', {})

comparison_data = [{
    'Model': 'Gradient Boosting',
    'Feature_Count': len(feature_names),
    'R2_Log': gb_metrics.get('r2_log', 0.4383),
    'RMSE_Log': gb_metrics.get('rmse_log', 1.1303),
    'RMSE_Original': gb_metrics.get('rmse_original', 2523.6),
    'MAE_Original': gb_metrics.get('mae_original', 1356.1)
}]

comparison_df = pd.DataFrame(comparison_data)
comparison_df['R2_Log'] = comparison_df['R2_Log'].round(4)
comparison_df['RMSE_Log'] = comparison_df['RMSE_Log'].round(4)
comparison_df['RMSE_Original'] = comparison_df['RMSE_Original'].round(1)
comparison_df['MAE_Original'] = comparison_df['MAE_Original'].round(1)

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

best_r2 = comparison_df.loc[0, 'R2_Log']
st.success(f"🏆 Model: **Gradient Boosting** with R² = **{best_r2:.4f}** on log scale (from notebook)")

# Feature importance (if available from saved model)
if hasattr(best_model, 'feature_importances_'):
    importance_values = best_model.feature_importances_
    if len(feature_names) == len(importance_values):
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance_df,
            x='Importance', y='Feature',
            orientation='h', title='Feature Importance - Gradient Boosting',
            color='Importance', color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)

st.caption("📊 Features selected via consensus ranking (Lasso + Random Forest + Gradient Boosting)")

# ==================== Interactive Price Prediction ====================
st.header("🎯 Interactive Price Predictor")

st.markdown("Adjust the phone specifications below to get an AI-powered price estimate")

col1, col2, col3 = st.columns(3)

# Default values based on data
brand_default = df['Brand'].mode()[0] if len(df) > 0 else "Apple"
storage_default = df['Storage_GB'].mode()[0] if len(df) > 0 else 256
condition_default = int(df['Condition_Score'].median()) if len(df) > 0 else 85

with col1:
    brand_input = st.selectbox("Brand", options=df['Brand'].unique(), index=0)
    storage_input = st.selectbox("Storage (GB)", options=sorted(df['Storage_GB'].unique()), 
                                  index=sorted(df['Storage_GB'].unique()).index(256) if 256 in df['Storage_GB'].unique() else 0)
    condition_input = st.slider("Condition Score (0-100)", 0, 100, condition_default)

with col2:
    months_input = st.slider("Months Since Release", 0, 60, 12)
    original_price_input = st.number_input("Original Price (HKD)", min_value=1000, max_value=20000, value=7999, step=100)
    product_origin_unknown = st.selectbox("Product Origin Unknown", ["No", "Yes"])

with col3:
    is_premium_tier = st.selectbox("Premium Tier", ["No", "Yes"])
    days_since_posted = st.slider("Days Since Posted", 0, 365, 7)
    warranty_input = st.selectbox("Has Warranty?", ["No", "Yes"])

# Prepare input features for prediction
def prepare_prediction_input():
    """Prepare input features - EXACTLY matching selected_features order"""
    input_dict = {
        'Months_Since_Release': months_input,
        'Condition_Score': condition_input,
        'Storage_GB': storage_input,
        'Product_Origin_Unknown': 1 if product_origin_unknown == "Yes" else 0,
        'Is_Premium_Tier': 1 if is_premium_tier == "Yes" else 0,
        'Post_Recency_Log': np.log1p(days_since_posted),
        'Days_Since_Posted': days_since_posted,
        'Original_Price_HKD': original_price_input,
        'Brand_Apple': 1 if brand_input == "Apple" else 0
    }
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])[feature_names]
    
    return input_df

input_features = prepare_prediction_input()
input_scaled = scaler.transform(input_features)

# Get prediction from saved model
pred_log = best_model.predict(input_scaled)[0]
pred_original = np.expm1(pred_log)

# Display prediction
st.subheader("📈 AI Price Prediction")

st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; border-radius: 1rem; text-align: center;">
    <h3 style="color: white;">🎯 Gradient Boosting Prediction</h3>
    <h1 style="color: white; font-size: 3rem;">HK${pred_original:,.0f}</h1>
    <p style="color: white;">R² (log scale): {best_r2:.4f} | MAE: HK${gb_metrics.get('mae_original', 1356):,.0f}</p>
</div>
""", unsafe_allow_html=True)

# Price range based on MAE
mae = gb_metrics.get('mae_original', 1356)
st.info(f"📊 Predicted price range (±MAE): HK${pred_original - mae:,.0f} ~ HK${pred_original + mae:,.0f}")

# ==================== AI LLM Integration ====================
st.header("🤖 AI Assistant - Smart Recommendations")

st.markdown("Get personalized buying/selling recommendations from our AI assistant")

with st.expander("📝 AI Analysis & Recommendations", expanded=True):
    st.markdown("### 📊 Market Analysis")
    
    if len(filtered_df) > 0 and len(filtered_df[filtered_df['Brand']==brand_input]) > 0:
        col1, col2 = st.columns(2)
        with col1:
            # Price position analysis
            brand_avg = filtered_df[filtered_df['Brand']==brand_input]['Price'].mean()
            if pred_original > brand_avg * 1.1:
                st.info(f"📈 This device is priced **{((pred_original/brand_avg)-1)*100:.0f}% above** the average {brand_input} price")
            elif pred_original < brand_avg * 0.9:
                st.success(f"📉 This device is priced **{(1 - pred_original/brand_avg)*100:.0f}% below** the average {brand_input} price")
            else:
                st.info(f"📊 This device is priced **around market average** for {brand_input}")
        
        with col2:
            # Condition impact
            condition_impact = filtered_df.groupby('Brand')['Condition_Score'].mean()
            if condition_input > condition_impact.get(brand_input, 80):
                st.success(f"✨ Above-average condition (+{condition_input - condition_impact.get(brand_input, 80):.0f} points) → Higher value")
            elif condition_input < condition_impact.get(brand_input, 80):
                st.warning(f"⚠️ Below-average condition ({condition_input - condition_impact.get(brand_input, 80):.0f} points) → Lower value")
            else:
                st.info("✅ Average condition for this brand")
    else:
        st.info("📊 Market data available for analysis")
    
    st.markdown("### 💡 Smart Recommendations")
    
    recommendations = []
    
    if condition_input < 70:
        recommendations.append("🔧 Consider professional cleaning/repair before selling to increase value")
    if warranty_input == "Yes" and months_input < 12:
        recommendations.append("📋 Device has warranty - highlight this in your listing")
    if len(filtered_df) > 0:
        if pred_original > filtered_df['Price'].quantile(0.75):
            recommendations.append("💰 Premium pricing range - target collectors or brand enthusiasts")
        elif pred_original < filtered_df['Price'].quantile(0.25):
            recommendations.append("🎯 Great value deal - quick sale expected if priced competitively")
    
    if not recommendations:
        recommendations.append("✅ Fairly priced device with good market positioning")
    
    for rec in recommendations:
        st.write(rec)
    
    # Analysis based on prediction
    st.markdown("### 🧠 AI Deep Analysis")
    
    if pred_original > 8000:
        analysis = "This is a premium-tier device. The high price reflects strong brand value and likely good condition. Consider highlighting unique features and providing detailed photos to justify the price point."
    elif pred_original > 4000:
        analysis = "Mid-range pricing detected. This segment is highly competitive. Emphasize battery health, included accessories, and any remaining warranty to stand out from similar listings."
    else:
        analysis = "Budget-friendly pricing. Focus on value proposition - good condition, functional device at an accessible price point. Quick sale expected in this segment."
    
    st.info(f"💭 {analysis}")
    
    # Selling tips
    st.markdown("### 📝 Selling Tips")
    
    avg_images = int(filtered_df['image_count'].mean()) if len(filtered_df) > 0 else 5
    tips = [
        f"📸 Include {max(5, avg_images)}+ high-quality photos",
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
    img_array = np.array(image.convert('L'))  # Convert to grayscale
    
    # Calculate image quality metrics
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    sharpness = np.std(np.diff(img_array, axis=0)) + np.std(np.diff(img_array, axis=1))
    
    # Detect potential defects based on image texture
    from scipy import ndimage
    
    # Edge detection (may indicate scratches or cracks)
    edges = ndimage.sobel(img_array)
    edge_density = np.mean(np.abs(edges) > 30)
    
    # Uniformity detection (may indicate worn areas)
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
    
    # Screen condition assessment
    if features['edge_density'] > 0.15:
        screen_condition = "⚠️ Potential scratches/cracks detected"
        screen_score = 60
    elif features['edge_density'] > 0.08:
        screen_condition = "📱 Minor signs of use"
        screen_score = 75
    else:
        screen_condition = "✨ Screen in excellent condition"
        screen_score = 90
    
    # Body condition assessment
    if features['uniformity'] < 0.6:
        body_condition = "🔄 Visible wear on body"
        body_score = 65
    elif features['uniformity'] < 0.75:
        body_condition = "👍 Normal signs of use"
        body_score = 80
    else:
        body_condition = "🌟 Body looks like new"
        body_score = 95
    
    # Image quality assessment
    if features['sharpness'] < 50:
        image_quality = "📷 Image quality is low, consider retaking the photo"
    else:
        image_quality = "✅ Image is clear, accurate assessment possible"
    
    # Overall score
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
    try:
        # Load and display image
        image = Image.open(uploaded_file)
        
        # Resize image for display
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
                # Extract features
                features = extract_image_features(image)
                # Analyze condition
                analysis = analyze_phone_condition(image, features)
            
            # Display analysis results
            st.markdown(f"""
            <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                <p><strong>📱 Screen Condition:</strong> {analysis['screen_condition']}</p>
                <p><strong>🔧 Body Condition:</strong> {analysis['body_condition']}</p>
                <p><strong>{analysis['image_quality']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar for overall score
            st.progress(analysis['overall_score'] / 100)
            st.metric("📊 Overall Condition Score", f"{analysis['overall_score']}/100")
            
            # Recommendations based on score
            if analysis['overall_score'] >= 85:
                st.success("✅ Phone is in excellent condition! Can be listed as 'Like New' or 'Minor signs of use'")
                price_adjustment = "+5%"
            elif analysis['overall_score'] >= 70:
                st.info("👍 Phone is in good condition with normal wear, price at market average")
                price_adjustment = "0%"
            else:
                st.warning("⚠️ Phone shows significant wear, consider lowering price for faster sale")
                price_adjustment = "-10%"
            
            # Price adjustment recommendation
            st.caption(f"💡 Based on image analysis, suggested price adjustment: {price_adjustment}")
            
            # Optional: Show detailed image metrics (for debugging)
            with st.expander("📊 Detailed Image Metrics"):
                st.json({
                    "Brightness": f"{features['brightness']:.1f}",
                    "Contrast": f"{features['contrast']:.1f}",
                    "Sharpness": f"{features['sharpness']:.1f}",
                    "Edge Density": f"{features['edge_density']:.3f}",
                    "Uniformity": f"{features['uniformity']:.3f}"
                })
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
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

if len(filtered_df) > 0:
    col1, col2 = st.columns(2)

    with col1:
        # Price trend by storage
        storage_trend = filtered_df.groupby('Storage_GB')['Price'].agg(['mean', 'count']).reset_index()
        storage_trend = storage_trend[storage_trend['count'] > 5]
        
        if len(storage_trend) > 0:
            fig_trend = px.line(
                storage_trend, x='Storage_GB', y='mean',
                title='Price Trend by Storage Capacity',
                markers=True,
                labels={'mean': 'Average Price (HKD)', 'Storage_GB': 'Storage (GB)'}
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        # Condition vs price by brand
        filtered_df['Condition_Label'] = pd.cut(
            filtered_df['Condition_Score'], 
            bins=[0, 70, 85, 100], 
            labels=['Poor', 'Good', 'Excellent']
        )
        brand_condition = filtered_df.groupby(['Brand', 'Condition_Label']).size().unstack().fillna(0)
        
        if len(brand_condition) > 0:
            fig_condition_stack = px.bar(
                brand_condition.reset_index(),
                x='Brand', y=['Poor', 'Good', 'Excellent'],
                title='Condition Distribution by Brand',
                barmode='stack',
                color_discrete_sequence=['#ff6b6b', '#ffd93d', '#6bcb77']
            )
            st.plotly_chart(fig_condition_stack, use_container_width=True)

    # Depreciation analysis
    st.subheader("📉 Value Depreciation Analysis")

    depreciation = filtered_df.groupby('Months_Since_Release')['Value_Retention'].mean().reset_index()
    depreciation = depreciation[depreciation['Months_Since_Release'] <= 60]

    if len(depreciation) > 0:
        fig_depreciation = px.scatter(
            depreciation, x='Months_Since_Release', y='Value_Retention',
            title='Value Retention Over Time',
            trendline='lowess',
            labels={'Months_Since_Release': 'Months Since Release', 
                    'Value_Retention': 'Value Retention (%)'}
        )
        fig_depreciation.update_traces(marker=dict(size=8, color='#667eea'))
        st.plotly_chart(fig_depreciation, use_container_width=True)

# ==================== Footer ====================
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### 📱 System Info
- **Total Listings:** {len(df):,}
- **Brands:** {df['Brand'].nunique()}
- **Avg Price:** HK${df['Price'].mean():,.0f}
- **Best Model:** Gradient Boosting
- **R² Score:** {best_r2:.4f}
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🚀 Features
- 🤖 AI-Powered Price Prediction
- 📸 Image-based Condition Analysis
- 🧠 Smart Recommendations
- 📊 Real-time Market Analytics
""")

# Main execution
if __name__ == "__main__":
    pass
