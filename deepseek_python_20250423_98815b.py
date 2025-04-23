import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("🛒 Market Basket Analysis - Recommendation Engine")

# Load default dataset
@st.cache_data
def load_default_data():
    return pd.read_csv("Home_store.csv")

# Function to perform market basket analysis
def perform_analysis(df):
    # Convert dataset to transactions
    transactions = df.apply(lambda row: [item for item in row if pd.notna(item)], axis=1).tolist()
    
    # Transaction Encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Generate Frequent Itemsets
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
    
    return transactions, rules

# Use default dataset or uploaded file
df = None
uploaded_file = st.file_uploader("Upload your own transaction dataset (CSV - optional)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()
    st.info("Using default Home Store dataset. You can upload your own file above if needed.")

st.subheader("📊 Data Preview")
st.dataframe(df.head())

# Perform analysis
transactions, rules = perform_analysis(df)

# Get all unique items from the dataset
all_items = sorted(set(item for transaction in transactions for item in transaction))

# UI Elements
st.subheader("🔍 Get Product Recommendations")

# Create two columns - one for the dropdown, one for the button
col1, col2 = st.columns([3, 1])

with col1:
    selected_item = st.selectbox(
        "Select a product:",
        all_items,
        index=0,
        key="item_select",
        help="Select a product to see what items are frequently bought with it"
    )

with col2:
    st.write("")  # Empty space for alignment
    st.write("")  # Empty space for alignment
    get_rec = st.button("Get Recommendations")

if get_rec:
    # Filter rules where selected item is in antecedents
    recommended = rules[rules['antecedents'].apply(lambda x: selected_item in x)]
    
    if not recommended.empty:
        # Sort by lift and get top 10
        recommended = recommended.sort_values(by='lift', ascending=False).head(10)
        
        # Format the output
        recommended_display = recommended[['antecedents', 'consequents', 'confidence', 'lift']].copy()
        recommended_display['antecedents'] = recommended_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        recommended_display['consequents'] = recommended_display['consequents'].apply(lambda x: ', '.join(list(x)))
        
        st.subheader(f"✨ Recommended items to buy with {selected_item}")
        st.dataframe(recommended_display[['antecedents', 'consequents', 'confidence', 'lift']])
        
        # Show some statistics
        st.markdown(f"**Strongest recommendation:** {recommended_display.iloc[0]['consequents']} (lift: {recommended_display.iloc[0]['lift']:.2f})")
    else:
        st.warning(f"No strong recommendations found for {selected_item}. Try lowering the support threshold if you see this often.")