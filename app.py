from flask import Flask, render_template, jsonify, request
from testingcode import NewsRecommender
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Recommender
print("Initializing Recommender System...")
recommender = NewsRecommender()
print("Recommender Initialized.")

# Default static user data
DEFAULT_USER_DATA = {
    "categories": ["Technology", "Sports"],
    "likes": [
        {"text": "Latest Marvel movie breaks box office records with huge opening weekend.", "timestamp": datetime.now()},
        {"text": "Review of the new Christopher Nolan film: A masterpiece of cinema.", "timestamp": datetime.now()},
        {"text": "Grammy awards winners announced: Taylor Swift takes home album of the year.", "timestamp": datetime.now()}
    ],
    "history": [
        {"text": "NBA Finals: Lakers secure victory in thrilling overtime game.", "timestamp": datetime.now()},
        {"text": "Manchester United transfer rumors: Top striker looking to move.", "timestamp": datetime.now()},
        {"text": "Netflix announces new season of hit fantasy series.", "timestamp": datetime.now()},
        {"text": "Formula 1 race results: Verstappen wins again.", "timestamp": datetime.now()}
    ]
}

# --- HELPER FUNCTION: This does the logic for both Home Page and API ---
def get_recommendations_logic(user_input=None):
    # Prepare User Data
    current_data = DEFAULT_USER_DATA.copy()
    
    if user_input:
        if 'categories' in user_input:
             current_data['categories'] = user_input['categories']
        if 'custom_likes' in user_input and user_input['custom_likes']:
            lines = [l.strip() for l in user_input['custom_likes'].split('\n') if l.strip()]
            current_data['likes'] = [{"text": l, "timestamp": datetime.now()} for l in lines]
        if 'custom_history' in user_input and user_input['custom_history']:
            lines = [l.strip() for l in user_input['custom_history'].split('\n') if l.strip()]
            current_data['history'] = [{"text": l, "timestamp": datetime.now()} for l in lines]

    # 1. Update Recommender Data
    recommender.user_data = current_data
    
    # 2. Generate User Vector
    recommender.generate_user_embedding()
    
    # 3. Get Recommendations (Articles)
    articles = recommender.recommend_articles(num_recommendations=20)
    
    # 4. Generate Summary
    summary_data = recommender.generate_briefing(articles)
    
    # 5. Format Articles
    formatted_articles = []
    for i, art in enumerate(articles, 1):
        formatted_articles.append({
            "id": i,
            "title": art.get("title", "No title"),
            "body": art.get("body", "")[:200] + "...",
            "category": art.get("category", "General"),
            "score": round(art.get("recommendation_score", 0), 2),
            "date": art.get("date", str(datetime.now().date())),
            "url": art.get("url", "#")
        })

    return {
        "status": "success", 
        "summary": summary_data, 
        "articles": formatted_articles
    }

# --- ROUTES ---

@app.route('/')
def home():
    # Run the logic IMMEDIATELY when the page loads
    initial_data = get_recommendations_logic(user_input=None)
    # Pass this data to the HTML template
    return render_template('index.html', initial_data=initial_data)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json
        data = get_recommendations_logic(user_input)
        return jsonify(data)
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)