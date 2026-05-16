import matplotlib.pyplot as plt
import seaborn as sns

# Ensure segments DataFrame exists from previous pipeline
# segments = pd.read_csv("promo_campaign_recommendations.csv")

# -------------------------------
# 1. Promo Response Rate by City Tier
# -------------------------------
plt.figure(figsize=(8,5))
sns.barplot(x=segments[cols.get("city")], y=segments["Promo_Response_Rate"], ci=None, palette="viridis")
plt.title("Promo Response Rate by City Tier")
plt.ylabel("Response Rate")
plt.xlabel("City Tier")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 2. Average Spend by Age Group
# -------------------------------
plt.figure(figsize=(8,5))
sns.barplot(x=segments[cols.get("age")], y=segments["Avg_Spend"], ci=None, palette="magma")
plt.title("Average Monthly Spend by Age Group")
plt.ylabel("Avg Spend (INR)")
plt.xlabel("Age Group")
plt.show()

# -------------------------------
# 3. Recommended Campaign Distribution
# -------------------------------
plt.figure(figsize=(10,6))
sns.countplot(y=segments["Recommended_Campaign"], palette="coolwarm")
plt.title("Distribution of Recommended Campaigns")
plt.xlabel("Number of Segments")
plt.ylabel("Campaign Type")
plt.show()

# -------------------------------
# 4. ROI Simulation (Promo vs Non-Promo)
# -------------------------------
segments["Simulated_Revenue"] = segments["Avg_Spend"] * segments["Avg_Frequency"]
segments["Promo_Revenue"] = np.where(segments["Promo_Response_Rate"] > 0.3,
                                     segments["Simulated_Revenue"] * 1.2,  # 20% uplift if campaign succeeds
                                     segments["Simulated_Revenue"])

plt.figure(figsize=(8,5))
sns.barplot(x=["Baseline Revenue", "Promo Revenue"],
            y=[segments["Simulated_Revenue"].sum(), segments["Promo_Revenue"].sum()],
            palette="Set2")
plt.title("Simulated ROI Uplift from Promo Campaigns")
plt.ylabel("Total Revenue (INR)")
plt.show()
