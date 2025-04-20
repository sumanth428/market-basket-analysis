import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title("🛒 Market Basket Analysis - Instacart Style")

uploaded_file = st.file_uploader("Upload your instacart CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Group transactions by member + date
    df['Date'] = pd.to_datetime(df['Date'])
    transactions = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list)
    transaction_list = transactions.tolist()

    # One-hot encoding
    te = TransactionEncoder()
    te_array = te.fit(transaction_list).transform(transaction_list)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    # Apriori
    frequent_items = apriori(df_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1.0)

    # Show frequent itemsets
    st.subheader("📦 Frequent Itemsets")
    st.dataframe(frequent_items.sort_values("support", ascending=False).head(10))

    # Show rules
    st.subheader("🔗 Association Rules")
    rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    st.dataframe(rules_display.sort_values("confidence", ascending=False).head(10))

    # Download button
    csv = rules.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Rules as CSV",
        data=csv,
        file_name='instacart_rules.csv',
        mime='text/csv',
    )
