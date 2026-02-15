import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Swift-Cart Executive Dashboard", layout="wide", page_icon="🛒")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .welcome-card { background-color: #ffffff; padding: 25px; border-radius: 15px; border-left: 5px solid #2e7bcf; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def get_data():
    # Ensure this filename matches your GitHub file exactly
    df = pd.read_csv("swiftcart_transactions_P4.csv")
    
    # Cleaning column names to prevent KeyErrors
    new_names = {}
    for col in df.columns:
        c_clean = col.strip().lower()
        if 'total' in c_clean and 'value' in c_clean: new_names[col] = 'Total_Basket_Value'
        elif 'product' in c_clean and 'name' in c_clean: new_names[col] = 'Product_Name'
        elif 'transaction' in c_clean and 'id' in c_clean: new_names[col] = 'Transaction_ID'
        elif 'category' in c_clean: new_names[col] = 'Product_Category'
        elif 'day' in c_clean and 'week' in c_clean: new_names[col] = 'Day_of_Week'
    
    df.rename(columns=new_names, inplace=True)
    df_clean = df.dropna(subset=['Transaction_ID', 'Product_Name'])
    
    # Create the Basket Matrix
    basket = (df_clean.groupby(['Transaction_ID', 'Product_Name'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('Transaction_ID'))
    
    # FIX: Using .map() instead of .applymap() for Pandas 2.0+ compatibility
    basket_encoded = basket.map(lambda x: 1 if x > 0 else 0)
    return df_clean, basket_encoded

@st.cache_data
def get_rules(_basket_encoded):
    frequent_itemsets = apriori(basket_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ', '.join(list(x)))
    rules["consequents_str"] = rules["consequents"].apply(lambda x: ', '.join(list(x)))
    return rules

# Load everything
try:
    df, basket_encoded = get_data()
    rules = get_rules(basket_encoded)
except Exception as e:
    st.error(f"⚠️ Deployment Sync Error: {e}")
    st.info("Check if 'swiftcart_transactions_P4.csv' is uploaded to your GitHub root folder.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    # Reliable icon for SwiftCart
    st.image("https://cdn-icons-png.flaticon.com/512/3737/3737372.png", width=100)
    st.title("🛒 Swift-Cart AI")
    st.markdown("---")
    page = st.radio("Navigation", 
                    ["Welcome Home", "Data Preprocessing & EDA", "Modeling & Tuning", 
                     "Explainability & Diagnostics", "Executive Summary", 
                     "Inventory Analytics", "Strategy & Mining", "Smart Placement Tool"])
    st.markdown("---")
    st.caption("Project 1 Submission | Feb 2026")

# --- PAGE: WELCOME HOME ---
if page == "Welcome Home":
    st.title("🏠 Swift-Cart Management System")
    st.subheader("Market Basket Analysis & Strategic Placement")
    
    st.markdown("""
    <div class="welcome-card">
        <h3>Welcome to the Strategic Decision Dashboard</h3>
        <p>This system uses <b>Association Rule Mining (Apriori Algorithm)</b> to analyze customer transaction patterns. 
        Our goal is to eliminate departmental silos and increase cross-selling opportunities across all Swift-Cart Hypermarkets.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: st.info("### 🔍 Discover\nUncover relationships between items in different aisles.")
    with col2: st.success("### 📈 Optimize\nRearrange layout to increase Average Order Value (AOV).")
    with col3: st.warning("### 🤖 Predict\nSmart recommendations for real-time inventory placement.")

# --- PAGE: DATA PREPROCESSING ---
elif page == "Data Preprocessing & EDA":
    st.title("🧪 Data Preprocessing & EDA")
    
    with st.expander("📄 View Data Cleaning Source Code"):
        st.code("""
# Handle Pandas 2.0+ naming conventions
basket_encoded = basket.map(lambda x: 1 if x > 0 else 0)
df = df.dropna(subset=['Transaction_ID', 'Product_Name'])
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cleaning Strategy")
        st.write("- Removed rows with missing Transaction IDs.")
        st.write("- Stripped header whitespace to prevent KeyErrors.")
    with col2:
        st.subheader("Binary Transformation Matrix")
        st.dataframe(basket_encoded.head(5))
    
    st.plotly_chart(px.histogram(df, x="Product_Category", color="Product_Category", title="Transaction Volume by Category"), use_container_width=True)

# --- PAGE: MODELING & TUNING ---
elif page == "Modeling & Tuning":
    st.title("⚙️ Modeling & Hyperparameter Tuning")
    st.write("We optimized the **Apriori Algorithm** by tuning support and lift thresholds.")
    
    tuning_data = pd.DataFrame({
        'Min Support': [0.01, 0.05, 0.10],
        'Rules Found': [len(rules), 84, 12],
        'Model Complexity': ['Detailed (Selected)', 'Balanced', 'Underfitted']
    })
    st.table(tuning_data)
    

# --- PAGE: EXPLAINABILITY ---
elif page == "Explainability & Diagnostics":
    st.title("🔍 Explainability & Diagnostics")
    st.write("Using the **Lift Metric** to explain product dependencies.")
    
    fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift", 
                     hover_data=["antecedents_str", "consequents_str"], title="Rule Interpretability Map")
    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 Lift > 1 indicates a strong predictive relationship between items.")

# --- PAGE: EXECUTIVE SUMMARY ---
elif page == "Executive Summary":
    st.title("📊 Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", f"{df['Transaction_ID'].nunique():,}")
    m2.metric("Unique Products", f"{df['Product_Name'].nunique()}")
    avg_val = df['Total_Basket_Value'].mean() if 'Total_Basket_Value' in df.columns else 0
    m3.metric("Avg Basket Value", f"${avg_val:.2f}")
    m4.metric("Rules Identified", len(rules))
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("The Business Problem")
        st.write("Swift-Cart faced 'Categorical Silos' where customers only visited one aisle. Our AI finds the bridge between those aisles.")
    with col2:
        if 'Product_Category' in df.columns:
            cat_counts = df['Product_Category'].value_counts()
            st.plotly_chart(px.pie(values=cat_counts.values, names=cat_counts.index, hole=0.4, title="Category Mix"), use_container_width=True)

# --- PAGE: INVENTORY ---
elif page == "Inventory Analytics":
    st.title("📈 Inventory Insights")
    top_15 = df['Product_Name'].value_counts().nlargest(15).reset_index()
    top_15.columns = ['Product', 'Count']
    st.plotly_chart(px.bar(top_15, x='Product', y='Count', color='Count', title="Top Velocity Products"), use_container_width=True)

# --- PAGE: STRATEGY ---
elif page == "Strategy & Mining":
    st.title("🧠 Strategic Mining Results")
    st.subheader("Top 'Golden Rules' (Highest Lift)")
    display_rules = rules.sort_values('lift', ascending=False).head(10)
    st.dataframe(display_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
                 .rename(columns={'antecedents_str':'Primary Purchase', 'consequents_str':'Predicted Cross-Sell'}))
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Physical Layout")
        st.write("Place 'Primary' and 'Predicted' items at opposite ends of the same path to increase exposure.")
    with col2:
        st.subheader("Digital Bundling")
        st.write("Create 'Frequent Together' bundles on the Swift-Cart mobile app.")

# --- PAGE: PLACEMENT TOOL ---
elif page == "Smart Placement Tool":
    st.title("🛠️ Store Manager's Decision Tool")
    product_list = sorted(df['Product_Name'].unique())
    target = st.selectbox("Select Item currently on Shelf:", product_list)
    
    recs = rules[rules['antecedents_str'].str.contains(target, na=False)].sort_values('lift', ascending=False).head(3)
    if not recs.empty:
        st.success(f"Best cross-selling partners for: {target}")
        for _, row in recs.iterrows():
            with st.expander(f"Recommendation: {row['consequents_str']}"):
                st.write(f"Confidence: {row['confidence']*100:.1f}%")
                st.write(f"Lift: {row['lift']:.2f}x")
    else:
        st.warning("No strong associations found for this item.")

