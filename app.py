"""
ForgeX - Document Forgery Detection Web Application
===================================================
Beautiful web interface for document forgery detection with real-time analysis.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
from pathlib import Path
import sys
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

# Add src to path
sys.path.append('src/features')
from build_features import DocumentFeatureExtractor

# Page configuration
st.set_page_config(
    page_title="ForgeX - Document Forgery Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .upload-text {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 0.5rem 0;
    }
    .result-authentic {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-forged {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .feature-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class ForgeryDetectorApp:
    """Main application class for forgery detection"""
    
    def __init__(self):
        """Initialize the detector with trained models"""
        self.model_loaded = False
        self.model = None
        self.scaler = None
        self.selector = None
        self.extractor = DocumentFeatureExtractor()
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing components"""
        try:
            model_path = Path('models/final_ensemble_model.joblib')
            scaler_path = Path('models/final_scaler.joblib')
            selector_path = Path('models/final_feature_selector.joblib')
            
            if model_path.exists() and scaler_path.exists() and selector_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.selector = joblib.load(selector_path)
                self.model_loaded = True
            else:
                st.error("⚠️ Model files not found. Please train the model first.")
                st.info("Run: `python train_final_model.py` to train the model.")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
    
    def extract_enhanced_features(self, image):
        """Extract enhanced features from image"""
        # Convert PIL to numpy array
        img = np.array(image)
        
        # Save temporarily to extract features
        temp_path = 'temp_image.jpg'
        cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Get base features
        features = self.extractor.extract_all_features(temp_path)
        if not features:
            return None
        
        # Add enhanced features
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # DCT features
        dct = cv2.dct(np.float32(gray))
        features['dct_mean'] = np.mean(dct)
        features['dct_std'] = np.std(dct)
        features['dct_max'] = np.max(np.abs(dct))
        
        # Compression artifacts
        _, compressed = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 
                                      [cv2.IMWRITE_JPEG_QUALITY, 90])
        decompressed = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
        decompressed_rgb = cv2.cvtColor(decompressed, cv2.COLOR_BGR2RGB)
        diff = cv2.absdiff(img, decompressed_rgb)
        features['compression_diff_mean'] = np.mean(diff)
        features['compression_diff_std'] = np.std(diff)
        
        # Noise analysis
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        high_freq = cv2.filter2D(gray, -1, kernel)
        features['noise_level'] = np.mean(np.abs(high_freq))
        features['noise_variance'] = np.var(high_freq)
        
        # Edge consistency
        edges = cv2.Canny(gray, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        features['edge_consistency'] = np.sum(dilated) / (gray.shape[0] * gray.shape[1])
        
        # Contrast variation
        block_size = 32
        h, w = gray.shape
        contrasts = []
        for i in range(0, h-block_size, block_size):
            for j in range(0, w-block_size, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                contrasts.append(np.std(block))
        features['contrast_variation'] = np.std(contrasts) if contrasts else 0
        features['contrast_mean'] = np.mean(contrasts) if contrasts else 0
        
        # Color histogram entropy
        hist_b = cv2.calcHist([img], [0], None, [256], [0,256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0,256])
        hist_r = cv2.calcHist([img], [2], None, [256], [0,256])
        
        def entropy(hist):
            hist = hist / (hist.sum() + 1e-7)
            return -np.sum(hist * np.log2(hist + 1e-7))
        
        features['color_hist_entropy'] = (entropy(hist_b) + entropy(hist_g) + entropy(hist_r)) / 3
        
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)
        
        return list(features.values())
    
    def predict(self, image):
        """Predict if image is authentic or forged"""
        if not self.model_loaded:
            return None
        
        # Extract features
        features = self.extract_enhanced_features(image)
        if features is None:
            return None
        
        # Preprocess
        features = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        features_selected = self.selector.transform(features_scaled)
        
        # Predict
        prediction = self.model.predict(features_selected)[0]
        probabilities = self.model.predict_proba(features_selected)[0]
        
        result = {
            'prediction': 'Authentic' if prediction == 0 else 'Forged',
            'confidence': max(probabilities) * 100,
            'probabilities': {
                'Authentic': probabilities[0] * 100,
                'Forged': probabilities[1] * 100
            },
            'risk_level': self.get_risk_level(max(probabilities) * 100, prediction)
        }
        
        return result
    
    def get_risk_level(self, confidence, prediction):
        """Determine risk level based on confidence and prediction"""
        if prediction == 0:  # Authentic
            if confidence > 90:
                return "Very Low Risk"
            elif confidence > 75:
                return "Low Risk"
            else:
                return "Medium Risk - Review Recommended"
        else:  # Forged
            if confidence > 90:
                return "Critical - High Confidence Forgery"
            elif confidence > 75:
                return "High Risk - Likely Forgery"
            else:
                return "Medium Risk - Possible Forgery"

def create_gauge_chart(confidence, prediction):
    """Create a gauge chart for confidence visualization"""
    color = "#38ef7d" if prediction == "Authentic" else "#ff6a00"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level", 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_probability_chart(probabilities):
    """Create bar chart for probability distribution"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Authentic', 'Forged'],
            y=[probabilities['Authentic'], probabilities['Forged']],
            marker=dict(
                color=['#38ef7d', '#ff6a00'],
                line=dict(color='rgba(0,0,0,0.5)', width=2)
            ),
            text=[f"{probabilities['Authentic']:.1f}%", f"{probabilities['Forged']:.1f}%"],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probability Distribution",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[0, 100]),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14)
    )
    
    return fig

def main():
    """Main application function"""
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = ForgeryDetectorApp()
    
    detector = st.session_state.detector
    
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #667eea;'>
            🔍 ForgeX - Document Forgery Detection
        </h1>
        <p style='text-align: center; font-size: 1.2rem; color: #666;'>
            AI-Powered Document Authentication System | 94.94% Accuracy
        </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/security-checked.png", width=80)
        st.title("About ForgeX")
        st.markdown("""
        **ForgeX** is an advanced AI-powered system for detecting forged documents.
        
        ### 🎯 Key Features
        - **94.94% Accuracy** on test data
        - **Real-time Analysis** in seconds
        - **32 Advanced Features** analyzed
        - **Ensemble ML Model** for reliability
        
        ### 📊 Model Performance
        - **Precision:** 97.24%
        - **Recall:** 92.50%
        - **ROC AUC:** 98.22%
        
        ### 🔬 Technology
        - Random Forest
        - Gradient Boosting
        - SVM & Logistic Regression
        - Advanced Feature Engineering
        """)
        
        st.markdown("---")
        st.markdown("**Developer:** Janhvi Aditi")
        st.markdown("**GitHub:** [@JanhviAditi](https://github.com/JanhviAditi)")
    
    # Check if model is loaded
    if not detector.model_loaded:
        st.error("⚠️ Model not loaded. Please train the model first.")
        st.info("Run: `python train_final_model.py`")
        return
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Document Analysis", "📁 Batch Processing", "📊 Analytics"])
    
    with tab1:
        st.markdown("### Upload Document for Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload an Aadhaar card or similar document image"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Document", use_column_width=True)
                
                # Analyze button
                if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
                    with st.spinner("Analyzing document... Please wait"):
                        result = detector.predict(image)
                        
                        if result:
                            st.session_state.result = result
                            st.session_state.analyzed_image = image
                        else:
                            st.error("❌ Error analyzing document. Please try another image.")
        
        with col2:
            if 'result' in st.session_state:
                result = st.session_state.result
                
                # Display result
                if result['prediction'] == 'Authentic':
                    st.markdown(f"""
                        <div class="result-authentic">
                            ✅ AUTHENTIC DOCUMENT
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result-forged">
                            ⚠️ FORGED DOCUMENT DETECTED
                        </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Confidence", f"{result['confidence']:.2f}%")
                with col_b:
                    st.metric("Risk Level", result['risk_level'])
                
                # Gauge chart
                st.plotly_chart(
                    create_gauge_chart(result['confidence'], result['prediction']),
                    use_container_width=True
                )
                
                # Probability chart
                st.plotly_chart(
                    create_probability_chart(result['probabilities']),
                    use_container_width=True
                )
                
                # Detailed information
                with st.expander("📋 Detailed Analysis"):
                    st.write("**Probability Breakdown:**")
                    st.write(f"- Authentic: {result['probabilities']['Authentic']:.2f}%")
                    st.write(f"- Forged: {result['probabilities']['Forged']:.2f}%")
                    st.write(f"\n**Risk Assessment:** {result['risk_level']}")
                    st.write(f"\n**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Download report
                if st.button("📥 Download Report", use_container_width=True):
                    report = f"""
FORGEX - DOCUMENT ANALYSIS REPORT
{'='*50}

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULT: {result['prediction']}
Confidence: {result['confidence']:.2f}%
Risk Level: {result['risk_level']}

PROBABILITY DISTRIBUTION:
- Authentic: {result['probabilities']['Authentic']:.2f}%
- Forged: {result['probabilities']['Forged']:.2f}%

MODEL INFORMATION:
- Model: Ensemble (RF + GB + SVM + LR)
- Accuracy: 94.94%
- Features Analyzed: 32

{'='*50}
ForgeX - AI-Powered Document Authentication
Developer: Janhvi Aditi
GitHub: @JanhviAditi
                    """
                    st.download_button(
                        "Download Text Report",
                        report,
                        file_name=f"forgex_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
    
    with tab2:
        st.markdown("### Batch Document Processing")
        st.info("Upload multiple documents for batch analysis")
        
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple Aadhaar cards or document images"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} files uploaded**")
            
            if st.button("🔍 Analyze All Documents", type="primary"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {file.name}")
                    image = Image.open(file)
                    result = detector.predict(image)
                    
                    if result:
                        results.append({
                            'Filename': file.name,
                            'Result': result['prediction'],
                            'Confidence': f"{result['confidence']:.2f}%",
                            'Risk Level': result['risk_level'],
                            'Authentic Probability': f"{result['probabilities']['Authentic']:.2f}%",
                            'Forged Probability': f"{result['probabilities']['Forged']:.2f}%"
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("✅ Analysis complete!")
                
                # Display results
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        authentic_count = len([r for r in results if r['Result'] == 'Authentic'])
                        st.metric("Authentic Documents", authentic_count)
                    with col2:
                        forged_count = len([r for r in results if r['Result'] == 'Forged'])
                        st.metric("Forged Documents", forged_count)
                    with col3:
                        avg_confidence = np.mean([float(r['Confidence'].strip('%')) for r in results])
                        st.metric("Average Confidence", f"{avg_confidence:.2f}%")
                    
                    # Download batch results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Batch Results (CSV)",
                        csv,
                        file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    with tab3:
        st.markdown("### System Analytics & Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Model Performance")
            metrics_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'ROC AUC'],
                'Value': [94.94, 97.24, 92.50, 98.22]
            }
            fig = px.bar(metrics_data, x='Metric', y='Value', 
                        title="Model Performance Metrics (%)",
                        color='Value',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Feature Categories")
            features_data = {
                'Category': ['Statistical', 'Edge Detection', 'DCT', 'Compression', 
                            'Noise Analysis', 'Texture', 'Color'],
                'Count': [5, 4, 3, 2, 2, 8, 8]
            }
            fig = px.pie(features_data, values='Count', names='Category',
                        title="32 Features by Category")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("#### 🔬 Technical Specifications")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Model Architecture**
            - Random Forest (300 trees)
            - Gradient Boosting (200 est.)
            - SVM (RBF kernel)
            - Logistic Regression
            """)
        
        with col2:
            st.markdown("""
            **Training Data**
            - 8,000 total images
            - 4,000 authentic
            - 4,000 forged (synthetic)
            - 80/20 train-test split
            """)
        
        with col3:
            st.markdown("""
            **Feature Engineering**
            - 32 total features
            - DCT analysis
            - Compression artifacts
            - Edge consistency
            - Noise patterns
            """)

if __name__ == "__main__":
    main()
