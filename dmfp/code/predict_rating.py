"""
Deployment script for Amazon product rating prediction model
Used to predict the potential ratings of new products

Author: Jiacheng Gu
"""

import pandas as pd
import numpy as np
import joblib
import re
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class AmazonRatingPredictor:
    """Amazon Product Rating Predictor Class"""
    
    def __init__(self, model_dir='model_results', use_latest=True):
        """
        Initialize the predictor
        
        Args:
            model_dir: Model files directory
            use_latest: Whether to use the latest model files
        """
        self.model = None
        self.scaler = None
        self.tfidf = None
        self.label_encoder = None
        self.feature_names = None
        self.model_info = None
        self.model_type = None
        self.model_dir = model_dir
        self.load_model_components(use_latest)
    
    def find_latest_model_files(self):
        """Find the model files"""
        # Use fixed filenames
        return {
            'model': os.path.join(self.model_dir, 'best_model.pkl'),
            'scaler': os.path.join(self.model_dir, 'scaler.pkl'),
            'tfidf': os.path.join(self.model_dir, 'tfidf.pkl'),
            'label_encoder': os.path.join(self.model_dir, 'label_encoder.pkl'),
            'model_info': os.path.join(self.model_dir, 'model_info.pkl')
        }
    
    def load_model_components(self, use_latest):
        """Load model components"""
        try:
            if use_latest:
                files = self.find_latest_model_files()
                print(f"Using latest model files...")
            else:
                # Use fixed filenames
                files = {
                    'model': 'amazon_rating_prediction_model.pkl',
                    'scaler': 'amazon_rating_scaler.pkl',
                    'tfidf': 'amazon_rating_tfidf.pkl',
                    'label_encoder': 'amazon_rating_label_encoder.pkl',
                    'model_info': None
                }
            
            # Load model components
            self.model = joblib.load(files['model'])
            self.scaler = joblib.load(files['scaler'])
            self.tfidf = joblib.load(files['tfidf'])
            self.label_encoder = joblib.load(files['label_encoder'])
            
            # Load model info (if exists)
            if files['model_info'] and os.path.exists(files['model_info']):
                self.model_info = joblib.load(files['model_info'])
                self.model_type = self.model_info.get('best_model_name', 'Unknown')
                self.feature_names = self.model_info.get('feature_names', [])
                print(f"Model type: {self.model_type}")
            else:
                # Try to infer from model object
                model_class_name = type(self.model).__name__
                self.model_type = model_class_name
                print(f"Inferred model type: {self.model_type}")
            
            print("Model components loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"Model file not found: {e}")
            print("Please make sure you've run the training script and generated model files")
            print(f"Search directory: {self.model_dir}")
            
            # List available model files
            if os.path.exists(self.model_dir):
                pkl_files = glob.glob(os.path.join(self.model_dir, '*.pkl'))
                if pkl_files:
                    print("\nAvailable model files:")
                    for f in pkl_files:
                        print(f"  - {os.path.basename(f)}")
    
    def clean_price(self, price_str):
        """Clean price data"""
        if pd.isna(price_str):
            return 0
        price_clean = re.sub(r'[₹,]', '', str(price_str))
        try:
            return float(price_clean)
        except:
            return 0
    
    def clean_discount(self, discount_str):
        """Clean discount percentage data"""
        if pd.isna(discount_str):
            return 0
        discount_clean = re.sub(r'%', '', str(discount_str))
        try:
            return float(discount_clean)
        except:
            return 0
    
    def clean_rating_count(self, count_str):
        """Clean rating count data"""
        if pd.isna(count_str):
            return 0
        count_clean = re.sub(r'[,]', '', str(count_str))
        try:
            return int(count_clean)
        except:
            return 0
    
    def extract_features(self, product_data):
        """
        Extract features from product data
        
        Args:
            product_data: Dictionary containing product information
            
        Returns:
            Processed feature vector
        """
        # Clean price data
        discounted_price = self.clean_price(product_data.get('discounted_price', '0'))
        actual_price = self.clean_price(product_data.get('actual_price', '0'))
        discount_percentage = self.clean_discount(product_data.get('discount_percentage', '0'))
        rating_count = self.clean_rating_count(product_data.get('rating_count', '0'))
        
        # Calculate derived features
        price_ratio = discounted_price / (actual_price + 1)  # Avoid division by zero
        absolute_savings = actual_price - discounted_price
        
        # Text features
        product_name = product_data.get('product_name', '')
        about_product = product_data.get('about_product', '')
        category = product_data.get('category', '')
        
        product_name_length = len(product_name)
        about_product_length = len(about_product)
        category_depth = category.count('|') + 1 if category else 1
        
        # Main category
        main_category = category.split('|')[0] if category else 'Unknown'
        
        # Numeric features (maintain same order as in training)
        numeric_features = np.array([
            discounted_price, actual_price, discount_percentage,
            rating_count, price_ratio, absolute_savings,
            product_name_length, about_product_length, category_depth
        ])
        
        # Category feature encoding
        try:
            main_category_encoded = self.label_encoder.transform([main_category])[0]
        except:
            # If new category, use the encoding of the most common category
            main_category_encoded = 0
            print(f"Unknown category '{main_category}', using default encoding")
        
        # Text features (TF-IDF)
        text_features = self.tfidf.transform([about_product]).toarray()[0]
        
        # Combine all features (maintain same order as in training)
        features = np.concatenate([
            numeric_features,
            [main_category_encoded],
            text_features
        ])
        
        # Choose whether to standardize based on model type
        features = features.reshape(1, -1)
        
        if self.model_type in ['Linear Regression', 'LinearRegression']:
            # Use standardization for linear regression models
            features = self.scaler.transform(features)
        
        return features
    
    def predict_rating(self, product_data):
        """
        Predict product rating
        
        Args:
            product_data: Product information dictionary
            
        Returns:
            Predicted rating value
        """
        if self.model is None:
            print("Model not loaded, cannot make predictions")
            return None
        
        try:
            # Extract features
            features = self.extract_features(product_data)
            
            # Predict rating
            predicted_rating = self.model.predict(features)[0]
            
            # Ensure rating is within reasonable range
            predicted_rating = max(1.0, min(5.0, predicted_rating))
            
            return round(predicted_rating, 2)
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_with_confidence(self, product_data):
        """
        Predict rating and provide confidence analysis
        
        Args:
            product_data: Product information dictionary
            
        Returns:
            Dictionary containing predicted rating and analysis
        """
        predicted_rating = self.predict_rating(product_data)
        
        if predicted_rating is None:
            return None
        
        # Analyze influencing factors
        analysis = self.analyze_prediction_factors(product_data)
        
        return {
            'predicted_rating': predicted_rating,
            'rating_level': self.get_rating_level(predicted_rating),
            'analysis': analysis,
            'recommendations': self.get_recommendations(product_data, analysis),
            'model_type': self.model_type
        }
    
    def get_rating_level(self, rating):
        """Get rating level description"""
        if rating >= 4.5:
            return "Excellent (4.5-5.0)"
        elif rating >= 4.0:
            return "Good (4.0-4.5)"
        elif rating >= 3.5:
            return "Average (3.5-4.0)"
        elif rating >= 3.0:
            return "Below Average (3.0-3.5)"
        else:
            return "Poor (1.0-3.0)"
    
    def analyze_prediction_factors(self, product_data):
        """Analyze key factors influencing the prediction"""
        analysis = {}
        
        # Price analysis
        discounted_price = self.clean_price(product_data.get('discounted_price', '0'))
        actual_price = self.clean_price(product_data.get('actual_price', '0'))
        discount_percentage = self.clean_discount(product_data.get('discount_percentage', '0'))
        
        if discounted_price > 0:
            analysis['price_factor'] = "Reasonable price" if discounted_price < 1000 else "High-priced product"
        
        if discount_percentage > 50:
            analysis['discount_factor'] = "High discount advantage"
        elif discount_percentage > 20:
            analysis['discount_factor'] = "Moderate discount"
        else:
            analysis['discount_factor'] = "Low discount"
        
        # Description analysis
        about_product = product_data.get('about_product', '')
        if len(about_product) > 500:
            analysis['description_factor'] = "Detailed product description"
        elif len(about_product) > 200:
            analysis['description_factor'] = "Basic product description"
        else:
            analysis['description_factor'] = "Minimal product description"
        
        return analysis
    
    def get_recommendations(self, product_data, analysis):
        """Provide improvement suggestions based on analysis"""
        recommendations = []
        
        # Suggestions based on description length
        about_product = product_data.get('about_product', '')
        if len(about_product) < 200:
            recommendations.append("Enrich product description content, detailing product features and benefits")
        
        # Suggestions based on pricing
        discount_percentage = self.clean_discount(product_data.get('discount_percentage', '0'))
        if discount_percentage < 10:
            recommendations.append("Consider adding an appropriate discount to increase product attractiveness")
        
        # Suggestions based on category
        category = product_data.get('category', '')
        if not category:
            recommendations.append("Ensure accurate product categorization to help users find the product")
        
        if not recommendations:
            recommendations.append("Product information is fairly complete, continue maintaining quality standards")
        
        return recommendations

def demo_prediction():
    """Demonstrate prediction functionality"""
    print("Amazon Product Rating Prediction Demo")
    print("=" * 50)
    
    # Create predictor instance
    predictor = AmazonRatingPredictor()
    
    if predictor.model is None:
        print("Model loading failed, cannot run demo")
        return
    
    # Sample product data
    sample_products = [
        {
            "product_name": "High-Quality Wireless Bluetooth Earphones",
            "category": "Electronics|Audio|Headphones",
            "discounted_price": "₹999",
            "actual_price": "₹1,999",
            "discount_percentage": "50%",
            "rating_count": "1,250",
            "about_product": "These wireless Bluetooth earphones use the latest Bluetooth 5.0 technology, providing exceptional sound quality. Features active noise cancellation, 35-hour battery life, and IPX7 waterproof rating. Suitable for sports, commuting, and everyday use. Package includes earphones, charging case, USB-C charging cable, and multiple sizes of ear tips."
        },
        {
            "product_name": "Basic Charging Cable",
            "category": "Electronics|Cables",
            "discounted_price": "₹99",
            "actual_price": "₹199",
            "discount_percentage": "50%",
            "rating_count": "50",
            "about_product": "Standard charging cable"
        }
    ]
    
    for i, product in enumerate(sample_products, 1):
        print(f"\nProduct {i}: {product['product_name']}")
        print("-" * 30)
        
        result = predictor.predict_with_confidence(product)
        
        if result:
            print(f"Predicted Rating: {result['predicted_rating']}/5.0")
            print(f"Rating Level: {result['rating_level']}")
            print(f"Model Used: {result['model_type']}")
            print(f"Key Factors: {', '.join(result['analysis'].values())}")
            print("Recommendations:")
            for rec in result['recommendations']:
                print(f"  • {rec}")
        else:
            print("Prediction failed")

def validate_with_real_data():
    """Validate model accuracy using real data"""
    print("Validating Model Accuracy with Real Data")
    print("=" * 60)
    
    # Create predictor instance
    predictor = AmazonRatingPredictor()
    
    if predictor.model is None:
        print("Model not loaded, cannot validate")
        return
    
    try:
        # Load original data
        print("Loading original dataset...")
        df = pd.read_csv('amazon.csv')
        print(f"Data loaded successfully, {len(df)} records")
        
        # Data preprocessing (consistent with training)
        def clean_rating(rating_str):
            if pd.isna(rating_str):
                return np.nan
            try:
                return float(str(rating_str))
            except:
                return np.nan
        
        df['rating'] = df['rating'].apply(clean_rating)
        df = df.dropna(subset=['rating'])
        
        # 增加测试样本数量并使用分层抽样
        # 允许用户选择测试样本的大小
        print("\nSelect test sample size:")
        print("1. Small (100 samples)")
        print("2. Medium (300 samples)")
        print("3. Large (500 samples)")
        print("4. Very Large (1000 samples)")
        print("5. Custom size")
        
        sample_choice = input("Enter your choice (1-5), or press Enter for default (300): ").strip()
        
        if sample_choice == '1':
            test_size = 100
        elif sample_choice == '2' or sample_choice == '':
            test_size = 300
        elif sample_choice == '3':
            test_size = 500
        elif sample_choice == '4':
            test_size = 1000
        elif sample_choice == '5':
            try:
                custom_size = int(input("Enter custom test size: ").strip())
                test_size = max(10, min(custom_size, len(df)))
            except:
                print("Invalid input, using default size (300)")
                test_size = 300
        else:
            print("Invalid choice, using default size (300)")
            test_size = 300
        
        # 限制测试样本最大数量为数据集大小
        test_size = min(test_size, len(df))
        
        # 分层抽样: 确保不同评分等级的样本都被包含
        print("\nSelect sampling method:")
        print("1. Random sampling")
        print("2. Stratified sampling by rating")
        
        stratify_choice = input("Enter your choice (1-2), or press Enter for default (2): ").strip()
        
        if stratify_choice == '1':
            # 随机抽样
            test_sample = df.sample(n=test_size, random_state=42)
            print(f"Using random sampling...")
        else:
            # 分层抽样
            # 创建评分分组
            df['rating_group'] = pd.cut(df['rating'], 
                                        bins=[0, 1.5, 2.5, 3.5, 4.5, 5.1], 
                                        labels=['1', '2', '3', '4', '5'],
                                        include_lowest=True)
            
            # 计算每个评分组应该的样本数
            rating_counts = df['rating_group'].value_counts()
            print("\nRating distribution in dataset:")
            for rating, count in rating_counts.items():
                print(f"  Rating {rating}: {count} samples ({count/len(df)*100:.1f}%)")
            
            # 从每个评分组抽样
            test_sample = pd.DataFrame()
            for rating in rating_counts.index:
                group_df = df[df['rating_group'] == rating]
                # 计算这个评分组应该抽取的样本数量
                group_size = min(int(test_size * (rating_counts[rating] / len(df))), len(group_df))
                if group_size > 0:
                    group_sample = group_df.sample(n=group_size, random_state=42)
                    test_sample = pd.concat([test_sample, group_sample])
            
            print(f"Using stratified sampling by rating...")
        
        print(f"Selected {len(test_sample)} samples for validation...")
        print("-" * 40)
        
        # 实现进度显示
        from tqdm import tqdm
        import time
        
        predictions = []
        actual_ratings = []
        errors = []
        display_limit = min(20, len(test_sample))  # 显示前20个样本结果
        
        print(f"Processing {len(test_sample)} samples (showing first {display_limit} results)...")
        
        try:
            # 尝试使用tqdm进度条
            for idx, (_, row) in enumerate(tqdm(test_sample.iterrows(), total=len(test_sample), desc="Predicting")):
                # Prepare product data
                product_data = {
                    'product_name': row.get('product_name', ''),
                    'category': row.get('category', ''),
                    'discounted_price': row.get('discounted_price', ''),
                    'actual_price': row.get('actual_price', ''),
                    'discount_percentage': row.get('discount_percentage', ''),
                    'rating_count': row.get('rating_count', ''),
                    'about_product': row.get('about_product', '')
                }
                
                # Make prediction
                predicted_rating = predictor.predict_rating(product_data)
                actual_rating = row['rating']
                
                if predicted_rating is not None:
                    predictions.append(predicted_rating)
                    actual_ratings.append(actual_rating)
                    error = abs(predicted_rating - actual_rating)
                    errors.append(error)
                    
                    # Display limited prediction results
                    if idx < display_limit:
                        print(f"Sample {idx+1:2d}: Actual={actual_rating:.1f} | Predicted={predicted_rating:.1f} | Error={error:.2f} | {row['product_name'][:30]}...")
        except ImportError:
            # 如果没有tqdm，使用简单的进度显示
            print("Processing samples...")
            total = len(test_sample)
            for idx, (_, row) in enumerate(test_sample.iterrows()):
                if idx % 10 == 0:
                    print(f"Progress: {idx}/{total} ({idx/total*100:.1f}%)")
                
                # Prepare product data
                product_data = {
                    'product_name': row.get('product_name', ''),
                    'category': row.get('category', ''),
                    'discounted_price': row.get('discounted_price', ''),
                    'actual_price': row.get('actual_price', ''),
                    'discount_percentage': row.get('discount_percentage', ''),
                    'rating_count': row.get('rating_count', ''),
                    'about_product': row.get('about_product', '')
                }
                
                # Make prediction
                predicted_rating = predictor.predict_rating(product_data)
                actual_rating = row['rating']
                
                if predicted_rating is not None:
                    predictions.append(predicted_rating)
                    actual_ratings.append(actual_rating)
                    error = abs(predicted_rating - actual_rating)
                    errors.append(error)
                    
                    # Display limited prediction results
                    if idx < display_limit:
                        print(f"Sample {idx+1:2d}: Actual={actual_rating:.1f} | Predicted={predicted_rating:.1f} | Error={error:.2f} | {row['product_name'][:30]}...")
        
        if len(predictions) > 0:
            # Calculate performance metrics
            predictions = np.array(predictions)
            actual_ratings = np.array(actual_ratings)
            errors = np.array(errors)
            
            mae = np.mean(errors)  # Mean Absolute Error
            mse = np.mean(errors**2)  # Mean Squared Error
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            
            # Calculate R² score
            ss_res = np.sum((actual_ratings - predictions) ** 2)
            ss_tot = np.sum((actual_ratings - np.mean(actual_ratings)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate accuracy at different tolerance levels
            accuracy_01 = np.mean(errors <= 0.1) * 100
            accuracy_02 = np.mean(errors <= 0.2) * 100
            accuracy_03 = np.mean(errors <= 0.3) * 100
            accuracy_05 = np.mean(errors <= 0.5) * 100
            
            # 增加错误分析
            error_distribution = {
                '0.0-0.1': np.sum((errors > 0.0) & (errors <= 0.1)),
                '0.1-0.2': np.sum((errors > 0.1) & (errors <= 0.2)),
                '0.2-0.3': np.sum((errors > 0.2) & (errors <= 0.3)),
                '0.3-0.5': np.sum((errors > 0.3) & (errors <= 0.5)),
                '0.5-1.0': np.sum((errors > 0.5) & (errors <= 1.0)),
                '>1.0': np.sum(errors > 1.0)
            }
            
            print(f"\nModel Performance Evaluation Results:")
            print("=" * 40)
            print(f"Basic Metrics:")
            print(f"   Mean Absolute Error (MAE): {mae:.3f}")
            print(f"   Root Mean Square Error (RMSE): {rmse:.3f}")
            print(f"   R² Score: {r2:.3f}")
            print(f"   Test Sample Size: {len(predictions)}")
            
            print(f"\nPrediction Accuracy:")
            print(f"   Accuracy within ±0.1 points: {accuracy_01:.1f}%")
            print(f"   Accuracy within ±0.2 points: {accuracy_02:.1f}%")
            print(f"   Accuracy within ±0.3 points: {accuracy_03:.1f}%")
            print(f"   Accuracy within ±0.5 points: {accuracy_05:.1f}%")
            
            print(f"\nError Distribution:")
            total_samples = len(predictions)
            for error_range, count in error_distribution.items():
                print(f"   Error {error_range}: {count} samples ({count/total_samples*100:.1f}%)")
            
            print(f"\nPrediction Distribution:")
            print(f"   Actual rating range: {actual_ratings.min():.1f} - {actual_ratings.max():.1f}")
            print(f"   Predicted rating range: {predictions.min():.1f} - {predictions.max():.1f}")
            print(f"   Average error: {mae:.3f} points")
            print(f"   Maximum error: {errors.max():.3f} points")
            print(f"   Minimum error: {errors.min():.3f} points")
            
            # 增加按评分分组的错误分析
            print(f"\nError Analysis by Rating:")
            # 创建评分组
            rating_groups = [1, 2, 3, 4, 5]
            for rating in rating_groups:
                # 获取特定评分的索引
                rating_indices = np.where((actual_ratings >= rating-0.5) & (actual_ratings < rating+0.5))[0]
                if len(rating_indices) > 0:
                    group_errors = errors[rating_indices]
                    group_mae = np.mean(group_errors)
                    group_accuracy = np.mean(group_errors <= 0.5) * 100
                    print(f"   Rating {rating}: MAE={group_mae:.3f}, Accuracy(±0.5)={group_accuracy:.1f}%, Samples={len(rating_indices)}")
            
            # Performance level assessment
            print(f"\nModel Performance Level:")
            if mae < 0.3:
                print("   Excellent (MAE < 0.3)")
            elif mae < 0.5:
                print("   Good (0.3 ≤ MAE < 0.5)")
            elif mae < 0.7:
                print("   Average (0.5 ≤ MAE < 0.7)")
            else:
                print("   Needs improvement (MAE ≥ 0.7)")
            
            # Provide improvement suggestions
            print(f"\nModel Optimization Suggestions:")
            if r2 < 0.5:
                print("   • Consider adding more features or using more complex models")
            if mae > 0.5:
                print("   • Check outlier handling and feature engineering")
            if accuracy_05 < 80:
                print("   • Consider using ensemble learning or deep learning methods")
            else:
                print("   • Model performance is good, consider deploying for use")
        
        else:
            print("All predictions failed, please check the model and data")
    
    except FileNotFoundError:
        print("Could not find the amazon.csv file, make sure the data file is in the current directory")
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

def interactive_prediction():
    """Interactive prediction functionality"""
    print("\nInteractive Product Rating Prediction")
    print("=" * 50)
    print("Please enter product information for rating prediction:")
    
    predictor = AmazonRatingPredictor()
    
    if predictor.model is None:
        print("Model not loaded, cannot make predictions")
        return
    
    # Get user input
    product_data = {}
    product_data['product_name'] = input("Product Name: ")
    product_data['category'] = input("Product Category (e.g., Electronics|Audio): ")
    product_data['discounted_price'] = input("Discounted Price (e.g., ₹999): ")
    product_data['actual_price'] = input("Original Price (e.g., ₹1999): ")
    product_data['discount_percentage'] = input("Discount Percentage (e.g., 50%): ")
    product_data['rating_count'] = input("Rating Count (e.g., 1000): ")
    product_data['about_product'] = input("Product Description: ")
    
    # Make prediction
    result = predictor.predict_with_confidence(product_data)
    
    if result:
        print(f"\nPrediction Results:")
        print(f"Predicted Rating: {result['predicted_rating']}/5.0")
        print(f"Rating Level: {result['rating_level']}")
        print(f"Model Used: {result['model_type']}")
        print(f"\nAnalysis:")
        for key, value in result['analysis'].items():
            print(f"  {key}: {value}")
        print(f"\nRecommendations:")
        for rec in result['recommendations']:
            print(f"  • {rec}")
    else:
        print("Prediction failed, please check input data")

def create_simple_model_files():
    """Create simple model files for compatibility"""
    pass  # Feature removed

if __name__ == "__main__":
    print("Amazon Product Rating Prediction System")
    print("=" * 60)
    
    while True:
        print("\nPlease select an operation:")
        print("1. Run Demo Prediction")
        print("2. Validate with Real Data")
        print("3. Interactive Prediction")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            demo_prediction()
        elif choice == '2':
            validate_with_real_data()
        elif choice == '3':
            interactive_prediction()
        elif choice == '4':
            print("Thank you for using the system!")
            break
        else:
            print("Invalid choice, please try again")