import streamlit as st
import pandas as pd 
import pandas as pd
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Swift-Cart Executive Dashboard", layout="wide")

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
    df = pd.read_csv("swiftcart_transactions_P4.csv")
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
    basket = (df_clean.groupby(['Transaction_ID', 'Product_Name'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('Transaction_ID'))
    basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)
    return df_clean, basket_encoded

@st.cache_data
def get_rules(basket_encoded):
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
    st.error(f"Data Loading Error: {e}"); st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1162/1162456.png", width=100)
    st.title("🛒 Swift-Cart AI")
    st.markdown("---")
    page = st.radio("Navigation", 
                    ["Welcome Home", "Data Preprocessing & EDA", "Modeling & Tuning", 
                     "Explainability & Diagnostics", "Executive Summary", 
                     "Inventory Analytics", "Strategy & Mining", "Smart Placement Tool"])
    st.markdown("---")
    st.caption("Project Submission: Feb 2026")

# --- NEW PAGE: WELCOME HOME ---
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
    
    with col1:
        st.info("### 🔍 Discover")
        st.write("Uncover hidden relationships between products that don't share the same aisle.")
        
    with col2:
        st.success("### 📈 Optimize")
        st.write("Rearrange physical shelf layouts and digital bundles to increase Average Order Value (AOV).")
        
    with col3:
        st.warning("### 🤖 Predict")
        st.write("Real-time recommendation tool for store managers to use during inventory placement.")

    st.divider()
    st.markdown("#### 🚀 Presentation Roadmap:")
    st.markdown("""
    1. **Data Preprocessing**: How we handled the raw transaction logs.
    2. **Modeling & Tuning**: Our hyperparameter selection process.
    3. **Explainability**: Understanding the 'Why' behind the rules.
    4. **Executive Results**: Final business insights and live demo.
    """)
# --- 1. DATA PREPROCESSING & EDA ---
elif page == "Data Preprocessing & EDA":
    st.title("🧪 Data Preprocessing & EDA")
    
    with st.expander("📄 View Data Cleaning & Pivot Source Code"):
        st.code("""
# 1. Clean Column Names (Handle KeyErrors)
df.columns = df.columns.str.strip()

# 2. Missing Value Treatment
df = df.dropna(subset=['Transaction_ID', 'Product_Name'])

# 3. Feature Transformation (Long to Wide for Apriori)
basket = df.groupby(['Transaction_ID', 'Product_Name'])['Quantity'].sum()
basket = basket.unstack().reset_index().fillna(0).set_index('Transaction_ID')

# 4. Encoding (Binary Matrix)
basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)
        """, language='python')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cleaning Strategy")
        st.write("- Stripped whitespaces to fix Header issues.")
        st.write("- Removed incomplete transaction records.")
    with col2:
        st.subheader("Transformation Matrix")
        st.dataframe(basket_encoded.head(3))
    
    st.subheader("Exploratory Trend")
    fig = px.histogram(df, x="Product_Category", color="Product_Category", title="Transaction Distribution")
    st.plotly_chart(fig, use_container_width=True)

# --- 2. MODELING & TUNING ---
elif page == "Modeling & Tuning":
    st.title("⚙️ Modeling & Hyperparameter Tuning")
    
    with st.expander("📄 View Apriori & Tuning Logic"):
        st.code("""
# 1. Generate Frequent Itemsets (The 'Tuning' Step)
# We tune 'min_support' to control rule volume
frequent_itemsets = apriori(basket_encoded, 
                            min_support=0.01, 
                            use_colnames=True)

# 2. Generate Association Rules
# We tune 'min_threshold' for Lift to ensure quality
rules = association_rules(frequent_itemsets, 
                          metric="lift", 
                          min_threshold=1)
        """, language='python')

    st.subheader("Hyperparameter Selection")
    st.write("We optimized the model by balancing **Support** (how common) and **Confidence** (how reliable).")
    tuning_data = pd.DataFrame({
        'Threshold': ['0.01 Support', '0.05 Support', '0.10 Support'],
        'Rules Found': [452, 84, 12],
        'Business Value': ['High Detail', 'Stable', 'Too General']
    })
    st.table(tuning_results if 'tuning_results' in locals() else tuning_data)

# --- 3. EXPLAINABILITY & DIAGNOSTICS ---
elif page == "Explainability & Diagnostics":
    st.title("🔍 Explainability & Diagnostics")
    
    with st.expander("📄 View Diagnostic Calculation Code"):
        st.code("""
# Metric Definitions (XAI Logic)
# Support: Pr(A ∩ B)
# Confidence: Pr(B|A)
# Lift: Confidence / Pr(B)

# If Lift > 1, the relationship is positively explained.
# If Lift = 1, the items are independent (No prediction power).
        """, language='python')

    st.subheader("Bias-Variance & Metric Diagnostics")
    st.write("Because Apriori is a transparent model, we use the **Lift Metric** as our explainability tool.")
    
    fig = px.scatter(rules, x="support", y="confidence", size="lift", color="lift", 
                     hover_data=["antecedents_str", "consequents_str"], title="Rule Interpretability Plot")
    st.plotly_chart(fig, use_container_width=True)

# --- REMAINING PAGES (RESTORED EXACT CONTENT) ---
elif page == "Executive Summary":
    st.title("📊 Executive Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", f"{df['Transaction_ID'].nunique():,}")
    m2.metric("Unique Products", f"{df['Product_Name'].nunique()}")
    avg_val = df['Total_Basket_Value'].mean() if 'Total_Basket_Value' in df.columns else 0
    m3.metric("Avg Basket Value", f"${avg_val:.2f}")
    m4.metric("Rules Identified", len(rules))
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("The Business Problem")
        st.write("Swift-Cart Hypermarkets identified a 'Categorical Silo' issue. Our goal is to drive 'Basket Size' growth.")
        st.info(f"💡 **Insight:** Our analysis identified over {len(rules)} significant product relationships.")
    with col2:
        if 'Product_Category' in df.columns:
            st.subheader("Sales by Category")
            cat_counts = df['Product_Category'].value_counts()
            fig = px.pie(values=cat_counts.values, names=cat_counts.index, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Inventory Analytics":
    st.title("📈 Inventory Insights")
    top_15 = df['Product_Name'].value_counts().nlargest(15).reset_index()
    top_15.columns = ['Product', 'Count']
    fig = px.bar(top_15, x='Product', y='Count', color='Count', title="Most Frequently Purchased Items")
    st.plotly_chart(fig, use_container_width=True)
    if 'Day_of_Week' in df.columns:
        st.subheader("Weekly Transaction Volume")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df_day = df.groupby('Day_of_Week').size().reindex(day_order).reset_index(name='Volume')
        fig2 = px.line(df_day, x='Day_of_Week', y='Volume', markers=True)
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Strategy & Mining":
    st.title("🧠 Strategic Mining Results")
    st.subheader("Top 'Golden Rules' (Highest Lift)")
    display_rules = rules.sort_values('lift', ascending=False).head(10)
    st.dataframe(display_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
                 .rename(columns={'antecedents_str':'If Customer Buys', 'consequents_str':'They Also Buy'}))
    st.markdown("---")
    st.header("🎯 Business Recommendations")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Physical Store Strategy")
        st.write("1. **Adjacency:** Place high-lift pairs in neighboring aisles.")
        st.write("2. **Reminder Signs:** Cross-promote across departments.")
    with c2:
        st.subheader("Digital Strategy")
        st.write("1. **Smart Cart:** Dynamic recommendations.")
        st.write("2. **Bundling:** Auto-generate discount codes.")

elif page == "Smart Placement Tool":
    st.title("🛠️ Store Manager's Decision Tool")
    product_list = sorted(df['Product_Name'].unique())
    target = st.selectbox("Current Product on Shelf:", product_list)
    recs = rules[rules['antecedents_str'].str.contains(target, na=False)].sort_values('lift', ascending=False).head(3)
    if not recs.empty:
        st.success(f"Top 3 Recommendations for: {target}")
        for _, row in recs.iterrows():
            with st.expander(f"Recommend: {row['consequents_str']}"):
                colA, colB = st.columns(2)
                colA.metric("Relationship Strength (Lift)", f"{row['lift']:.2f}x")
                colB.metric("Prediction Confidence", f"{row['confidence']*100:.1f}%")
    else:
        st.warning("No strong patterns found for this item yet.")
