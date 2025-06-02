import streamlit as st
import sqlite3
import random
import json
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential
import os

# ----- Database Setup ----- #
def init_db():
    conn = sqlite3.connect('restaurants.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS restaurants
                 (id INTEGER PRIMARY KEY, name TEXT, location TEXT, cuisine TEXT, 
                  capacity INTEGER, open_time TEXT, close_time TEXT)''')
                  
    c.execute('''CREATE TABLE IF NOT EXISTS reservations
                 (id INTEGER PRIMARY KEY, restaurant_id INTEGER, customer_name TEXT,
                  time TEXT, guests INTEGER, status TEXT)''')
    
    # Sample data
    if c.execute("SELECT COUNT(*) FROM restaurants").fetchone()[0] == 0:
        locations = ["MG Road", "Koramangala", "Whitefield", "Indiranagar", "HSR Layout"]
        cuisines = ["Italian", "Indian", "Chinese", "Mexican", "Vegan", "Japanese"]
        
        for i in range(1, 26):
            c.execute("INSERT INTO restaurants VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (i, f"Restaurant {i}", random.choice(locations), 
                      random.choice(cuisines), random.randint(15, 50), 
                      "11:00", "23:00"))
    
    conn.commit()
    return conn

import pdb
# ----- AI Agent Core ----- #
class RestaurantAgent:
    def __init__(self):
        # Initialize Llama-3 client with automatic fallback
        self.llm_enabled = True
        try:
            #self.client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.2")
            #self.client = InferenceClient("mistralai/Devstral-Small-2505")
            #self.client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct")
            self.client = InferenceClient(api_key= os.environ["HUGGINGFACE_TOKEN"],model = "meta-llama/Llama-3.3-70B-Instruct")
            #self.client = InferenceClient("meta-llama/Llama-3.3-70B-Instruct")
            #pdb.set_trace()
            completion = self.client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)
            #test_response = self.client.post(json={"inputs": "test", "parameters": {"max_new_tokens": 5}})
            test_response = completion.choices[0].message
            #if not test_response.ok:
            if not test_response:
                raise ConnectionError("API test failed")
            else:
                print(test_response)
        except Exception as e:
            print(f"LLM unavailable, falling back to mock: {e}")
            self.llm_enabled = False

        self.available_functions = {
            "get_recommendations": self.get_recommendations,
            "book_table": self.book_table,
            "modify_reservation": self.modify_reservation
        }

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def detect_intent_with_llm(self, query):
        """Real LLM intent detection with structured output"""
        prompt = f"""
        Analyze this restaurant reservation query and return JSON:
        {{
            "intent": "<book_table|modify_reservation|get_recommendations>",
            "parameters": {{
                "location": "<str>", 
                "cuisine": "<str>",
                "time": "<str>",
                "guests": <int>,
                "reservation_id": <int>
            }}
        }}
        Query: "{query}"
        """
        completion = self.client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt #"What is the capital of France?"
                }
            ],
        )
        response = completion.choices[0].message
        """
        ChatCompletionOutputMessage(role='assistant', content='The capital of France is Paris.', tool_calls=None) 
        response = self.client.post(
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.3  # Reduce randomness
                }
            },
            timeout=10
        )
        """   

        #if not response.ok:
        if not response:
            raise ValueError(f"API error: {response.status_code}")
            
        try:
            result = json.loads(response.text)
            if not isinstance(result, dict):
                raise ValueError("Invalid JSON format")
            return result.get("intent", "fallback_response"), result.get("parameters", {})
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response")

    def detect_intent(self, query):
        """Hybrid detection: Try LLM first, fallback to mock"""
        if self.llm_enabled:
            try:
                return self.detect_intent_with_llm(query)
            except Exception as e:
                print(f"LLM failed: {e}, using mock detection")
                
        # Fallback to rule-based
        query = query.lower()
        if any(word in query for word in ["book", "reserve", "table"]):
            return "book_table", {"time": "7 PM", "guests": 2}
        elif any(word in query for word in ["change", "modify", "update"]):
            return "modify_reservation", {}
        elif any(word in query for word in ["recommend", "suggest", "find"]):
            return "get_recommendations", {}
        else:
            return "fallback_response", {}

    def execute_function(self, intent, parameters):
        """Execute the appropriate function"""
        if intent == "get_recommendations":
            results = self.get_recommendations(
                location=parameters.get("location"),
                cuisine=parameters.get("cuisine")
            )
            return self.format_recommendations(results)
            
        elif intent == "book_table":
            return self.book_table(
                restaurant_id=random.randint(1, 25),  # In prod, lookup by name
                time=parameters.get("time", "7 PM"),
                guests=parameters.get("guests", 2)
            )
            
        elif intent == "modify_reservation":
            return self.modify_reservation(
                reservation_id=parameters.get("reservation_id"),
                new_time=parameters.get("new_time")
            )
        else:
            return "I couldn't understand. Try: 'Book a table for 4 in Koramangala'"

    # ----- Tool Functions ----- #
    def get_recommendations(self, location=None, cuisine=None):
        conn = sqlite3.connect('restaurants.db')
        c = conn.cursor()
        query = "SELECT * FROM restaurants WHERE 1=1"
        params = []
        if location:
            query += " AND location=?"
            params.append(location)
        if cuisine:
            query += " AND cuisine=?"
            params.append(cuisine)
        c.execute(query, params)
        return c.fetchall()
    
    def book_table(self, restaurant_id, time, guests):
        conn = sqlite3.connect('restaurants.db')
        c = conn.cursor()
        c.execute("SELECT name, capacity FROM restaurants WHERE id=?", (restaurant_id,))
        name, capacity = c.fetchone()
        
        if guests > capacity:
            return f"❌ {name} only has {capacity} seats available"
            
        c.execute("INSERT INTO reservations VALUES (NULL, ?, ?, ?, ?, ?)",
                 (restaurant_id, "Customer", time, guests, "confirmed"))
        conn.commit()
        return f"✅ Booked at {name} for {time} (ID: {c.lastrowid})"
    
    def modify_reservation(self, reservation_id, new_time):
        if not reservation_id:
            return "❌ Please provide reservation ID"
            
        conn = sqlite3.connect('restaurants.db')
        c = conn.cursor()
        c.execute("UPDATE reservations SET time=? WHERE id=?", (new_time, reservation_id))
        conn.commit()
        return f"✅ Updated reservation #{reservation_id} to {new_time}"
    
    def format_recommendations(self, restaurants):
        if not restaurants:
            return "No matching restaurants found."
        return "\n".join([f"• {r[1]} ({r[3]}) in {r[2]}" for r in restaurants[:3]] + ["Say 'book at [name]' to reserve!"])

# ----- Streamlit UI ----- #
def main():
    st.set_page_config(page_title="FoodieSpot AI", layout="wide")
    st.title("🍽️ FoodieSpot Reservation Agent")
    #st.caption("Powered by Llama-3 AI | Fallback to mock detection when API unavailable")
    st.caption("Powered by Mistral AI | Fallback to mock detection when API unavailable")
    
    init_db()
    agent = RestaurantAgent()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I can help book tables or modify reservations."}
        ]
    
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # User input
    if prompt := st.chat_input("Try: 'Book a table for 2 in Indiranagar'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Process query
        intent, params = agent.detect_intent(prompt)
        response = agent.execute_function(intent, params)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
