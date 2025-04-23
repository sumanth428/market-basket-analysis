import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("🛒 Market Basket Analysis - Recommendation Engine")

# Load the dataset directly
@st.cache_data
def load_data():
    return pd.read_csv("Home_store.csv")

df = load_data()

# Convert dataset to transactions
transactions = df.apply(lambda row: [item for item in row if pd.notna(item)], axis=1).tolist()

# Transaction Encoding
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Generate Frequent Itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

# Get all unique items from the dataset
all_items = sorted(set(item for transaction in transactions for item in transaction))

# UI - Simple dropdown and button
selected_item = st.selectbox(
    "Select a product to get recommendations:",
    all_items
)

if st.button("Get Recommendations"):
    # Filter rules for selected item
    recommended = rules[rules['antecedents'].apply(lambda x: selected_item in x)]
    
    if not recommended.empty:
        recommended = recommended.sort_values(by='lift', ascending=False).head(5)
        
        st.subheader(f"💡 Recommended items with {selected_item}")
        
        # Display as a clean list
        for _, row in recommended.iterrows():
            consequents = ", ".join(list(row['consequents']))
            st.write(f"- **{consequents}** (confidence: {row['confidence']:.2f}, lift: {row['lift']:.2f})")
    else:
        st.warning(f"No strong recommendations found for {selected_item}")
