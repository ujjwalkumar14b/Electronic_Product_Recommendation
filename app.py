from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
df = pd.read_csv("Electronic_Product_Recommend.csv", encoding="latin1")

users = sorted(df["UserID"].astype(str).unique().tolist())
train_df = joblib.load("model_products.pkl")
popular_products = joblib.load("model_popularity.pkl")
user_model = joblib.load("model_cf_user.pkl")
item_model = joblib.load("model_cf_item.pkl")
svd = joblib.load("model_svd.pkl")

# PRODUCT DETAILS
product_details = {}
for product in df["ProductID"].unique():
    product = str(product)

    product_details[product] = {
        "Average Rating": round(
            df[df["ProductID"] == product]["Rating"].mean(),
            2
        ),
        "Total Ratings": int(
            df[df["ProductID"] == product]["Rating"].count()
        )
    }

# POPULARITY RECOMMENDATION
def recommend_popular(top_n=10):
    result = popular_products.head(top_n).reset_index()
    return result["ProductID"].tolist()

# SVD RECOMMENDATION
def recommend_svd(user_id, top_n=10):
    rated_products = set(train_df.loc[train_df["UserID"] == user_id, "ProductID"])
    predictions = []

    for product in train_df["ProductID"].unique():
        if product not in rated_products:
            score = svd.predict(user_id, product).est
            predictions.append((product, score))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [i[0] for i in predictions[:top_n]]

# HYBRID RECOMMENDATION
def hybrid_recommend(user_id, top_n=10):
    svd_products = recommend_svd(user_id, 50)
    popular = set(popular_products.head(100).index)
    result = []

    for product in svd_products:
        if product in popular:
            result.append(product)

    return result[:top_n]

# HOME
@app.route("/")
def home():
    return render_template(
        "index.html",
        users=users,
        selected_user=None,
        popular_recommendations=[],
        svd_recommendations=[],
        hybrid_recommendations=[],
        product_details=product_details
    )

# RECOMMEND
@app.route("/recommend", methods=["POST"])
def recommend():
    selected_user = request.form.get("user")
    popular_recommendations = recommend_popular()
    svd_recommendations = recommend_svd(selected_user)
    hybrid_recommendations = hybrid_recommend(selected_user)

    return render_template(
        "index.html",
        users=users,
        selected_user=selected_user,
        popular_recommendations=popular_recommendations,
        svd_recommendations=svd_recommendations,
        hybrid_recommendations=hybrid_recommendations,
        product_details=product_details
    )

# RUN
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)