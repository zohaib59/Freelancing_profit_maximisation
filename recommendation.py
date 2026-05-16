# ============================================================
# OPTIONAL NLP BUSINESS RECOMMENDATION ENGINE
# ============================================================
# PURPOSE:
# ------------------------------------------------------------
# Converts technical analytics into business-friendly insights
# understandable by:
#
# ✅ Marketing Teams
# ✅ Business Stakeholders
# ✅ Non-Technical Managers
# ✅ Clients
# ✅ Executives
#
# OPTIONAL:
# ------------------------------------------------------------
# You can place this section at the END of the pipeline.
# ============================================================

# ============================================================
# NLP RECOMMENDATION FUNCTION
# ============================================================

def generate_business_recommendation(row):

    cluster = row["Customer_Cluster"]

    spend = row["Avg_Spend"]

    basket = row["Avg_Basket"]

    frequency = row["Avg_Frequency"]

    response = row["Promo_Response_Rate"]

    campaign = row["Recommended_Campaign"]

    # --------------------------------------------------------
    # HIGH VALUE CUSTOMERS
    # --------------------------------------------------------

    if (
        spend >= high_spend and
        response >= high_response
    ):

        return (
            f"This customer segment belongs to a high-value "
            f"premium audience with strong campaign response. "
            f"They spend approximately ₹{spend:,.0f} on average "
            f"and respond positively to promotional activities. "
            f"The recommended strategy is '{campaign}', focusing "
            f"on loyalty rewards, exclusive offers, and premium "
            f"shopping experiences to maximize long-term retention."
        )

    # --------------------------------------------------------
    # HIGH BASKET CUSTOMERS
    # --------------------------------------------------------

    elif basket >= high_basket:

        return (
            f"This customer group tends to purchase multiple "
            f"items together with an average basket size of "
            f"{basket:.1f}. The business should focus on "
            f"bundle offers, combo deals, and cross-selling "
            f"strategies using the '{campaign}' approach "
            f"to increase overall revenue."
        )

    # --------------------------------------------------------
    # FREQUENT CUSTOMERS
    # --------------------------------------------------------

    elif frequency >= high_frequency:

        return (
            f"This segment visits or purchases frequently "
            f"with an average frequency of {frequency:.1f}. "
            f"These are repeat customers who can be retained "
            f"through membership benefits, reward points, "
            f"and personalized engagement campaigns under "
            f"the '{campaign}' strategy."
        )

    # --------------------------------------------------------
    # MODERATE PROMO RESPONDERS
    # --------------------------------------------------------

    elif response >= 0.30:

        return (
            f"This customer segment shows moderate interest "
            f"in promotional campaigns with a response rate "
            f"of {response:.0%}. Seasonal campaigns, festival "
            f"offers, and limited-time discounts are likely "
            f"to improve engagement and conversion rates."
        )

    # --------------------------------------------------------
    # LOW ENGAGEMENT CUSTOMERS
    # --------------------------------------------------------

    else:

        return (
            f"This segment currently shows lower engagement "
            f"levels and lower promotional response rates. "
            f"The business should focus on awareness campaigns, "
            f"introductory discounts, and customer reactivation "
            f"strategies to improve participation and retention."
        )

# ============================================================
# APPLY NLP ENGINE
# ============================================================

segments["Business_Recommendation"] = (
    segments.apply(
        generate_business_recommendation,
        axis=1
    )
)

print("\nBusiness NLP Recommendation Engine Completed")

# ============================================================
# OPTIONAL: EXPORT BUSINESS REPORT
# ============================================================

business_report_columns = [
    col for col in [
        cols.get("city"),
        age_group_col,
        cols.get("gender"),
        "Avg_Spend",
        "Avg_Basket",
        "Avg_Frequency",
        "Promo_Response_Rate",
        "Recommended_Campaign",
        "Business_Recommendation"
    ]
    if col is not None
]

business_report = segments[
    business_report_columns
]

business_report.to_csv(
    "business_friendly_recommendations.csv",
    index=False
)

print(
    "\nBusiness-Friendly Report Saved:"
)

print(
    "business_friendly_recommendations.csv"
)

# ============================================================
# OPTIONAL SAMPLE OUTPUT
# ============================================================

print("\n================================================")
print("SAMPLE BUSINESS RECOMMENDATIONS")
print("================================================")

for i in range(min(3, len(segments))):

    print("\n--------------------------------------------")

    print(
        f"Campaign: "
        f"{segments.iloc[i]['Recommended_Campaign']}"
    )

    print(
        f"Recommendation:\n"
    )

    print(
        segments.iloc[i][
            "Business_Recommendation"
        ]
    )

print("\n================================================")
