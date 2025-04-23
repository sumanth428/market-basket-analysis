
import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("🛒 Market Basket Analysis - Recommendation Engine")

# Upload CSV
uploaded_file = st.file_uploader("Upload your transaction dataset (CSV)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Convert dataset to transactions
    transactions = df.apply(lambda row: [item for item in row if pd.notna(item)], axis=1).tolist()

    # Transaction Encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # Generate Frequent Itemsets
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

    # Get unique items from the antecedents
    all_items = sorted(set(i for itemset in rules['antecedents'] for i in itemset))
    selected_item = st.selectbox("Select an item to get recommendations:", all_items)

    if st.button("Get Recommendations"):
        recommended = rules[rules['antecedents'].apply(lambda x: selected_item in x)]
        recommended = recommended[['antecedents', 'consequents', 'confidence', 'lift']]
        recommended = recommended.sort_values(by='lift', ascending=False).head(10)

        if not recommended.empty:
            st.subheader("📌 Recommended Items")
            st.dataframe(recommended)
        else:
            st.info("No strong recommendations found for the selected item.")
