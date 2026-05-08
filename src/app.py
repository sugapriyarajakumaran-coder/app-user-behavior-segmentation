import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="App User Segmentation Dashboard",
    layout="wide"
)

# ---------------- DARK THEME STYLE ----------------
st.markdown("""
<style>

[data-testid="stMetric"] {
    background-color: #111111;
    border-radius: 10px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: white !important;
    font-size: 16px !important;
    font-weight: bold;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 30px !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Data/clustered_users.csv")

# ---------------- TITLE ----------------
st.title("📊 App User Behavior Segmentation Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filters")

selected_cluster = st.sidebar.multiselect(
    "Select Cluster",
    options=df['cluster_name'].unique(),
    default=df['cluster_name'].unique()
)

filtered_df = df[df['cluster_name'].isin(selected_cluster)]

# ---------------- KPI SECTION ----------------
st.subheader("📌 Cluster KPIs")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Users", len(filtered_df))
col2.metric("Avg Engagement", round(filtered_df['engagement_score'].mean(),2))
col3.metric("Avg Sessions", round(filtered_df['sessions_per_week'].mean(),2))
col4.metric("Avg Inactivity", round(filtered_df['days_since_last_login'].mean(),2))

# ---------------- TABS ----------------
tab1, tab2, tab3= st.tabs([
    "📊 Cluster Distribution",
    "🧠 PCA Visualization",
    "🔍 User Search",
])

# ---------------- TAB 1 ----------------
with tab1:

    st.subheader("Cluster Distribution")

    col1, col2 = st.columns(2)

    # BAR CHART
    with col1:
        cluster_counts = filtered_df['cluster_name'].value_counts()

        fig, ax = plt.subplots(figsize=(6,4))
        cluster_counts.plot(kind='bar', ax=ax)
        plt.xticks(rotation=0)

        st.pyplot(fig)

    # PIE CHART
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,6))
        cluster_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        ax2.set_ylabel('')

        st.pyplot(fig2)

# ---------------- TAB 2 ----------------
with tab2:

    st.subheader("PCA Cluster Visualization")

    # Select numerical columns
    numerical_cols = filtered_df.select_dtypes(include=['int64', 'float64']).drop(
        columns=['cluster', 'mbk_cluster', 'db_cluster', 'gmm_cluster'],
        errors='ignore'
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numerical_cols)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig3, ax3 = plt.subplots(figsize=(10,6))

    scatter = ax3.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=filtered_df['cluster'],
        cmap='viridis',
        alpha=0.6
    )

    ax3.set_xlabel("PCA 1")
    ax3.set_ylabel("PCA 2")
    ax3.set_title("User Segmentation using PCA")

    legend = ax3.legend(*scatter.legend_elements(), title="Clusters")
    ax3.add_artist(legend)

    st.pyplot(fig3)


# ---------------- TAB 3 ----------------
with tab3:

    st.subheader("🔍 User Search & Profile")

    selected_user = st.selectbox(
        "Select User ID",
        filtered_df['user_id']
    )

    result = filtered_df[
        filtered_df['user_id'] == selected_user
    ]

    if not result.empty:

        user = result.iloc[0]

        st.markdown("## 👤 User Profile")

        col1, col2, col3 = st.columns(3)

        col1.metric("Cluster", user['cluster_name'])
        col2.metric("Engagement Score", round(user['engagement_score'],2))
        col3.metric("Churn Risk", round(user['churn_risk_score'],2))



        st.markdown("---")

        # USER DETAILS
        st.markdown("### 📊 User Behavior")

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.write(f"**Sessions Per Week:** {user['sessions_per_week']}")
            st.write(f"**Avg Session Duration:** {user['avg_session_duration_min']}")
            st.write(f"**Daily Active Minutes:** {user['daily_active_minutes']}")
            st.write(f"**Pages Viewed:** {user['pages_viewed_per_session']}")

        with detail_col2:
            st.write(f"**Content Downloads:** {user['content_downloads']}")
            st.write(f"**Social Shares:** {user['social_shares']}")
            st.write(f"**Last Login Gap:** {user['days_since_last_login']} days")
            st.write(f"**Marketing Source:** {user['marketing_source']}")

        st.markdown("---")

        # AI RECOMMENDATION
        st.markdown(" Recommendation")

        if user['cluster_name'] == "High Users":

            st.success(
                "This is a highly engaged customer. Recommend loyalty rewards, premium subscriptions, and referral programs."
            )

        elif user['cluster_name'] == "Low Users":

            st.error(
                "This user shows low engagement. Recommend retention campaigns, discounts, and reminder notifications."
            )

        elif user['cluster_name'] == "Moderate Users":

            st.info(
                "This user has moderate engagement. Personalized recommendations and feature suggestions may improve activity."
            )

        else:

            st.warning(
                "This user interacts occasionally. Recommend onboarding improvements and engagement notifications."
            )

        st.markdown("---")

        # FULL DATA
        st.markdown("### 📄 Complete User Data")

        st.dataframe(result)
