import streamlit as st
import pandas as pd

# Load the association rules from CSV
rules = pd.read_csv('association_rules.csv')

# App Title
st.title('Market Basket Analysis')

# Show the raw data
st.subheader("Association Rules")
st.dataframe(rules)

# Download button
st.download_button(
    label="Download Association Rules as CSV",
    data=rules.to_csv(index=False),
    file_name='association_rules.csv',
    mime='text/csv',
)

# Optional: Filter based on minimum lift
st.sidebar.header("Filter Rules")
min_lift = st.sidebar.slider("Minimum Lift", 0.0, float(rules['lift'].max()), 1.0, 0.1)
filtered_rules = rules[rules['lift'] >= min_lift]
st.subheader(f"Rules with Lift >= {min_lift}")
st.dataframe(filtered_rules)
