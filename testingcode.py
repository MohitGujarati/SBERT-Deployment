import json
import os
import numpy as np
import time
import sys
from datetime import datetime, timedelta
from typing import List, Dict

# Machine Learning Imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


# --- NEW IMPORT FOR GENERATION ---
from transformers import pipeline

# Configure standard output for UTF-8 (prevents emoji/text crashes)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

class NewsRecommender:
    def __init__(self):
        print(" Loading SBERT model (all-MiniLM-L6-v2)...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2') 
        print(" SBERT model loaded successfully!\n")

        # --- MAKE SURE THIS BLOCK IS HERE ---
        print(" Loading Generative Model (google/flan-t5-small)...")
        try:
            self.summarizer = pipeline("text2text-generation", model="google/flan-t5-small")
            print(" Generative Model loaded successfully!\n")
        except Exception as e:
            print(f" Warning: Could not load Generative Model. Error: {e}")
            self.summarizer = None
        
        # Initialize with empty data (will be injected later)
        self.user_data = {"categories": [], "likes": [], "history": []}
        
        # Cache category profiles for detection
        self.category_profiles = self._get_category_profiles()
        self.category_embeddings = self.sbert_model.encode(list(self.category_profiles.values()))

        # Placeholder for user embedding
        self.user_embedding = None

    def _get_category_profiles(self) -> Dict[str, str]:
        """Returns the detailed mapping of category names to descriptions."""
        return {
            "Technology": "Artificial intelligence, machine learning, robotics, computer science, gadgets, software, internet, cybersecurity, and tech industry innovations.",
            "Business": "Markets, finance, startups, economic policies, investments, corporate strategies, entrepreneurship, and business news.",
            "Science": "Space exploration, research breakthroughs, biology, chemistry, physics, astronomy, and scientific discoveries.",
            "Health": "Medical research, healthcare innovations, fitness, nutrition, disease prevention, mental wellness, and public health.",
            "Politics": "Government, elections, international relations, policies, law, diplomacy, and political events.",
            "Sports": "Football, basketball, cricket, tennis, hockey, tournaments, athletes, scores, championships, leagues, youth sports, and competitive games.",
            "Entertainment": "Movies, TV shows, celebrities, music, theater, pop culture, streaming platforms, and entertainment industry.",
            "Environment": "Climate change, sustainability, renewable energy, wildlife conservation, environmental protection, and ecological issues.",
        }

    def generate_user_embedding(self):
        """
        Generates a weighted average vector based on the injected user_data.
        Weights: Categories (20%), Likes (60%), History (20% with time decay).
        """
        profiles = self._get_category_profiles()
        
        # A. Encode Categories
        cat_texts = [profiles.get(cat, "") for cat in self.user_data["categories"]]
        if not cat_texts: cat_texts = ["General news"]
        cat_embeddings = self.sbert_model.encode(cat_texts)
        avg_cat_emb = np.mean(cat_embeddings, axis=0)

        # B. Encode Likes (High Importance - 60%)
        if self.user_data["likes"]:
            texts = [item['text'] for item in self.user_data["likes"]]
            like_embeddings = self.sbert_model.encode(texts)
            avg_like_emb = np.mean(like_embeddings, axis=0)
        else:
            avg_like_emb = np.zeros_like(avg_cat_emb)

        # C. Encode History (With Time Decay - 20%)
        if self.user_data["history"]:
            texts = [item['text'] for item in self.user_data["history"]]
            hist_embs = self.sbert_model.encode(texts)
            
            weighted_hist_embs = []
            total_weight = 0
            
            for i, _ in enumerate(self.user_data["history"]):
                # Decay: 1.0 for newest, decreasing for older items
                weight = 1.0 / (1.0 + (0.1 * i)) 
                weighted_hist_embs.append(hist_embs[i] * weight)
                total_weight += weight
            
            avg_hist_emb = np.sum(weighted_hist_embs, axis=0) / total_weight
        else:
            avg_hist_emb = np.zeros_like(avg_cat_emb)

        # D. Weighted Fusion
        has_behavior = len(self.user_data["likes"]) > 0 or len(self.user_data["history"]) > 0
        
        if has_behavior:
            # 20% Category, 60% Likes, 20% History
            final_emb = (0.2 * avg_cat_emb) + (0.6 * avg_like_emb) + (0.2 * avg_hist_emb)
        else:
            final_emb = avg_cat_emb

        print(" Generated Weighted User Embedding.")
        self.user_embedding = final_emb
        return final_emb



    def preprocess_article(self, article):
        title = article.get("title", "")
        body = article.get("body", "")
        return f"{title} {body}".strip()

    def detect_category_fast(self, article_emb):
        cats = list(self.category_profiles.keys())
        sims = cosine_similarity([article_emb], self.category_embeddings).flatten()
        best_idx = np.argmax(sims)
        
        # Threshold for "General" category
        if sims[best_idx] < 0.15:
            return "General"
        
        return cats[best_idx]

    def maximal_marginal_relevance(self, article_embeddings, user_embedding, articles_metadata, diversity=0.3, top_n=25):
        """Applies MMR Ranking with Recency Boost."""
        
        # 1. Base Similarity
        user_sims = cosine_similarity([user_embedding], article_embeddings).flatten()
        
        # 2. Recency Boost
        current_time = datetime.now()
        for i, article in enumerate(articles_metadata):
            try:
                date_str = article.get("dateTime", "")
                if date_str:
                    # Clean and parse date
                    date_str = date_str.replace("Z", "")
                    # Handle varying formats if necessary, standard is ISO
                    pub_date = datetime.fromisoformat(date_str)
                    days_old = (current_time - pub_date).days
                    
                    # Decay: lose 1% relevance per day old, max 20% penalty
                    decay_factor = max(0.8, 1.0 - (days_old * 0.01)) 
                    user_sims[i] *= decay_factor
            except Exception:
                pass 

        selected_indices = []
        candidate_indices = list(range(len(article_embeddings)))
        
        for _ in range(min(top_n, len(article_embeddings))):
            best_mmr = -np.inf
            best_idx = -1
            
            for idx in candidate_indices:
                relevance = user_sims[idx]
                redundancy = 0
                if selected_indices:
                    candidate_vec = article_embeddings[idx].reshape(1, -1)
                    selected_vecs = article_embeddings[selected_indices]
                    redundancy = np.max(cosine_similarity(candidate_vec, selected_vecs))
                
                mmr = (1 - diversity) * relevance - (diversity * redundancy)
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
                
        return selected_indices

    # --- STATIC DATA FOR SIMULATION ---
    STATIC_ARTICLES = [
        {"title": "SpaceX launches new Starship rocket successfully", "body": "Elon Musk's company achieved another milestone in space exploration with the successful orbit of the massive Starship rocket.", "date": "2024-05-20", "url": "#", "category": "Science"},
        {"title": "NASA discovers potential life signs on Europa", "body": "Scientists analyzing data from the Jupiter probe have found strong evidence of organic compounds in the ice shell.", "date": "2024-05-21", "url": "#", "category": "Science"},
        {"title": "New AI model predicts weather with 99% accuracy", "body": "DeepMind and Google Research release a new transformer model that outperforms traditional meteorological simulations.", "date": "2024-05-22", "url": "#", "category": "Technology"},
        {"title": "Apple announces revolutionary VR headset features", "body": "The latest Vision Pro update introduces neural interfaces and seamless mixed reality integration.", "date": "2024-05-23", "url": "#", "category": "Technology"},
        {"title": "Stock markets hit all-time high amidst tech boom", "body": "The S&P 500 and Nasdaq surged today as investor confidence in AI-driven companies continues to grow.", "date": "2024-05-24", "url": "#", "category": "Business"},
        {"title": "Federal Reserve signals potential interest rate cuts", "body": "Economic indicators suggest inflation is cooling, prompting the Fed to consider easing monetary policy.", "date": "2024-05-25", "url": "#", "category": "Business"},
        {"title": "Manchester City wins the Premier League title again", "body": "In a dramatic final day, Man City secured the trophy with a 3-1 victory, continuing their dominance.", "date": "2024-05-20", "url": "#", "category": "Sports"},
        {"title": "LeBron James announces retirement plans", "body": "The NBA legend stated that the upcoming season might be his last, marking the end of an era.", "date": "2024-05-21", "url": "#", "category": "Sports"},
        {"title": "New study shows benefits of Mediterranean diet", "body": "Researchers followed 10,000 participants and found significant reduction in heart disease risk.", "date": "2024-05-22", "url": "#", "category": "Health"},
        {"title": "Breakthrough in cancer vaccine research", "body": "Early trials of an mRNA-based vaccine show promising results in treating melanoma.", "date": "2024-05-23", "url": "#", "category": "Health"},
        {"title": "Summer blockbuster movie breaks box office records", "body": "The new superhero film has grossed over $1 billion in its opening week, shattering expectations.", "date": "2024-05-24", "url": "#", "category": "Entertainment"},
        {"title": "Pop star announces surprise world tour", "body": "Fans went wild as the dates were dropped on social media for the upcoming global stadium tour.", "date": "2024-05-25", "url": "#", "category": "Entertainment"},
        {"title": "Global summit addresses climate change urgency", "body": "World leaders gathered to sign a new accord aimed at reducing carbon emissions by 50% by 2030.", "date": "2024-05-20", "url": "#", "category": "Environment"},
        {"title": "Quantum computer solves previously impossible equation", "body": "IBM's latest quantum processor successfully calculated complex molecular interactions in seconds.", "date": "2024-05-26", "url": "#", "category": "Science"},
        {"title": "Electric vehicle sales surpass gas cars for first time", "body": "Global EV adoption reaches tipping point as prices drop and charging infrastructure expands.", "date": "2024-05-27", "url": "#", "category": "Technology"},
        {"title": "Amazon announces drone delivery expansion", "body": "The retail giant will now deliver packages via autonomous drones to 50 new cities.", "date": "2024-05-28", "url": "#", "category": "Business"},
        {"title": "Olympics opening ceremony dazzles millions", "body": "The spectacular event featured cutting-edge technology and celebrated global unity.", "date": "2024-05-29", "url": "#", "category": "Sports"},
        {"title": "Mental health app reduces anxiety by 60% in study", "body": "New research validates the effectiveness of AI-powered therapy applications.", "date": "2024-05-30", "url": "#", "category": "Health"},
        {"title": "Streaming service announces original series lineup", "body": "Netflix reveals 20 new shows featuring A-list actors and directors for the fall season.", "date": "2024-05-31", "url": "#", "category": "Entertainment"},
        {"title": "Arctic ice levels reach unexpected recovery", "body": "Climate scientists observe surprising reversal in ice cap melting trends this year.", "date": "2024-06-01", "url": "#", "category": "Environment"},
        {"title": "Mars rover discovers ancient water reservoir", "body": "Perseverance finds geological evidence of massive underground lake that existed billions of years ago.", "date": "2024-06-02", "url": "#", "category": "Science"},
        {"title": "5G network coverage reaches 90% of urban areas", "body": "Telecommunications companies complete major infrastructure upgrades across major cities.", "date": "2024-06-03", "url": "#", "category": "Technology"},
        {"title": "Startup unicorn valued at $10 billion in new funding", "body": "AI-focused company raises record Series D round from Silicon Valley investors.", "date": "2024-06-04", "url": "#", "category": "Business"},
        {"title": "Tennis star wins historic Grand Slam", "body": "The underdog player claimed victory in an epic five-set final at Wimbledon.", "date": "2024-06-05", "url": "#", "category": "Sports"},
        {"title": "New Alzheimer's drug approved by FDA", "body": "Clinical trials show the medication can slow cognitive decline by up to 40%.", "date": "2024-06-06", "url": "#", "category": "Health"},
        {"title": "Music festival announces eco-friendly initiatives", "body": "Coachella goes carbon-neutral with solar power and zero-waste policies.", "date": "2024-06-07", "url": "#", "category": "Entertainment"},
        {"title": "Coral reef restoration project shows major success", "body": "Scientists report 75% recovery in previously damaged reef systems using new techniques.", "date": "2024-06-08", "url": "#", "category": "Environment"},
        {"title": "Astronomers detect mysterious radio signals from deep space", "body": "Fast radio bursts from distant galaxy puzzle researchers and spark new theories.", "date": "2024-06-09", "url": "#", "category": "Science"},
        {"title": "Social media platform introduces AI content moderation", "body": "New system promises to detect harmful content 95% faster than human moderators.", "date": "2024-06-10", "url": "#", "category": "Technology"},
        {"title": "Cryptocurrency market rebounds after regulatory clarity", "body": "Bitcoin surges 25% following new government guidelines for digital assets.", "date": "2024-06-11", "url": "#", "category": "Business"},
        {"title": "World Cup qualifying matches produce shocking upsets", "body": "Underdog teams advance as favorites stumble in dramatic elimination rounds.", "date": "2024-06-12", "url": "#", "category": "Sports"},
        
    ]

    def get_static_articles(self):
        """Returns the static pool of articles for simulation."""
        return self.STATIC_ARTICLES
    def generate_briefing(self, articles):
        """
        Generates a structured list of summaries for the top 3 articles.
        Returns: List[Dict] -> [{'id': 1, 'title': '...', 'summary': '...'}]
        """
        if not articles:
            return []

        # Check if the model actually loaded
        if self.summarizer is None:
            return [{"id": 0, "title": "System Error", "summary": "Summarization model failed to load."}]

        print("Generating summary for top 3 articles...")
        
        # 1. Select only the top 3 articles
        top_3 = articles[:3] 
        briefing_data = []  # We will store objects here, not strings

        # 2. Loop through each of the top 3
        for i, article in enumerate(top_3, 1):
            title = article.get('title', 'No Title')
            body = article.get('body', '')
            
            # Prepare text 
            input_text = f"summarize: {body[:600]}"
            
            try:
                # Generate summary
                results = self.summarizer(input_text, max_length=60, min_length=20, do_sample=False)
                summary_text = results[0]['generated_text']
            except Exception as e:
                print(f"Error summarizing article {i}: {e}")
                summary_text = "Error generating summary."
            
            # 3. STRUCTURED DATA: Append a dictionary
            briefing_data.append({
                "id": i,
                "title": title,
                "summary": summary_text
            })

        # 4. Return the list directly (Flask's jsonify will handle it automatically)
        return briefing_data

    def recommend_articles(self, num_recommendations=25):
        print(f"\n [STATIC] Using Static Article Pool ({len(self.STATIC_ARTICLES)} articles)...")
        all_articles = self.get_static_articles()

        # Deduplicate
        seen_titles, unique_articles = set(), []
        for a in all_articles:
            t = a.get("title", "")
            if t and t not in seen_titles:
                seen_titles.add(t)
                unique_articles.append(a)

        # Encode
        print(f" Encoding {len(unique_articles)} articles...")
        all_texts = [self.preprocess_article(a) for a in unique_articles]
        article_embeddings = self.sbert_model.encode(all_texts, batch_size=32, show_progress_bar=True)
        
        # Categorize (Only if category is missing or we want to re-verify)
        for i, article in enumerate(unique_articles):
            if "category" not in article:
                article["category"] = self.detect_category_fast(article_embeddings[i])

        # Rank (MMR + Recency)
        print(" Calculating MMR Ranking...")
        selected_indices = self.maximal_marginal_relevance(
            article_embeddings, 
            self.user_embedding, 
            unique_articles, 
            diversity=0.3, 
            top_n=num_recommendations
        )
        
        final_recommendations = [unique_articles[i] for i in selected_indices]
        
        # Add visual scores
        for i, idx in enumerate(selected_indices):
            score = cosine_similarity([self.user_embedding], [article_embeddings[idx]])[0][0]
            final_recommendations[i]["recommendation_score"] = float(score)
        
        return final_recommendations

    def format_recommendations(self, recommendations, filename="test_output.json"):
        formatted = []
        for i, art in enumerate(recommendations, 1):
            formatted.append({
                "id": i,
                "title": art.get("title", "No title"),
                "category": art.get("category", "General"),
                "score": round(art.get("recommendation_score", 0), 2),
                "date": art.get("dateTime", "")[:10]
            })
            
        # Print to console for verification
        print("\n" + "="*80)
        print(" FINAL RESULTS (Sorted by Relevance + Diversity)")
        print("="*80)
        for item in formatted:
            print(f" {item['id']:2}. [{item['category']:12}] {item['title'][:60]:<60} (Score: {item['score']})")

        return formatted

def main():
    # --- STATIC DATA CONFIGURATION ---
    # User Profile:
    # 1. Interests: Tech and Sports
    # 2. Likes: Entertainment
    # 3. History: Sports and Entertainment
    
    static_user_data = {
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

    print("="*80)
    print(f" STARTING TEST: Static User Profile (Tech/Sports + Ent/Sports History)")
    print("="*80)

    try:
        # Initialize Recommender
        recommender = NewsRecommender() 
        
        # INJECT STATIC DATA
        print(" [TEST] Injecting static user data...")
        recommender.user_data = static_user_data
        
        # GENERATE VECTOR
        recommender.generate_user_embedding()
        
        # RUN RECOMMENDATION
        # Fetching static recommendations
        articles = recommender.recommend_articles(num_recommendations=15)
        
        # SHOW RESULTS
        recommender.format_recommendations(articles)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()