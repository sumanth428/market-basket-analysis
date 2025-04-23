import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Cache everything to prevent recomputation
@st.cache_data
def load_data():
    df = pd.read_csv("Home_store.csv")
    transactions = df.apply(lambda row: [item for item in row if pd.notna(item)], axis=1).tolist()
    return transactions

@st.cache_data
def get_rules(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)  # Higher support for demo
    return association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

def main():
    st.title("🛒 Product Recommender")
    
    try:
        transactions = load_data()
        rules = get_rules(transactions)
        
        items = sorted({item for tran in transactions for item in tran})
        selected = st.selectbox("Select a product:", items)
        
        if st.button("Recommend"):
            recommendations = rules[rules['antecedents'].apply(lambda x: selected in x)]
            if not recommendations.empty:
                st.write("Top matches:")
                for i, row in recommendations.head(5).iterrows():
                    st.write(f"👉 {', '.join(row['consequents'])} (confidence: {row['confidence']:.2f})")
            else:
                st.warning("No recommendations found. Try lowering support in code.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
