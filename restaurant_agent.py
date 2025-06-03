import streamlit as st
import sqlite3
import random
import os, pdb
import json
from typing import List, Dict, Optional
from huggingface_hub import InferenceClient

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
    
    # Sample data - 25 restaurants across 5 locations
    locations = ["MG Road", "Koramangala", "Whitefield", "Indiranagar", "HSR Layout"]
    cuisines = ["Italian", "Indian", "Chinese", "Mexican", "Vegan", "Japanese"]
    
    if c.execute("SELECT COUNT(*) FROM restaurants").fetchone()[0] == 0:
        for i in range(1, 26):
            c.execute("INSERT INTO restaurants VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (i, 
                      f"Restaurant {i}",
                      random.choice(locations),
                      random.choice(cuisines),
                      random.randint(15, 50),  # Capacity
                      "11:00",  # Open time
                      "23:00"   # Close time
                     ))
    conn.commit()
    return conn

# ----- Tool Implementations ----- #
def get_restaurant_recommendations(location: str, cuisine: Optional[str] = None) -> List[Dict]:
    """Tool 1: Find restaurants by location/cuisine"""
    conn = sqlite3.connect('restaurants.db')
    c = conn.cursor()
    
    query = "SELECT * FROM restaurants WHERE location = ?"
    params = [location]
    
    if cuisine:
        query += " AND cuisine = ?"
        params.append(cuisine)
    
    c.execute(query, params)
    return [dict(zip(['id', 'name', 'location', 'cuisine', 'capacity', 'open_time', 'close_time'], row)) 
            for row in c.fetchall()]

def book_table(restaurant_id: int, time: str, guests: int) -> str:
    """Tool 2: Create a reservation"""
    conn = sqlite3.connect('restaurants.db')
    c = conn.cursor()
    
    # Check capacity
    c.execute("SELECT name, capacity FROM restaurants WHERE id = ?", (restaurant_id,))
    result = c.fetchone()
    if not result:
        return "Error: Restaurant not found"
    
    name, capacity = result
    if guests > capacity:
        return f"Error: {name} only has {capacity} seats available"
    
    # Create reservation
    c.execute("INSERT INTO reservations VALUES (NULL, ?, ?, ?, ?, ?)",
             (restaurant_id, "Customer", time, guests, "confirmed"))
    conn.commit()
    return f"✅ Booked at {name} for {guests} at {time} (ID: {c.lastrowid})"

def cancel_reservation(reservation_id: int) -> str:
    """Tool 3: Cancel an existing booking"""
    conn = sqlite3.connect('restaurants.db')
    c = conn.cursor()
    
    c.execute("DELETE FROM reservations WHERE id = ?", (reservation_id,))
    conn.commit()
    return f"✅ Cancelled reservation #{reservation_id}" if c.rowcount > 0 else "❌ Reservation not found"

# ----- Tool Schemas for LLM ----- #
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_restaurant_recommendations",
            "description": "Find restaurants by location and optional cuisine type",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Neighborhood or area"},
                    "cuisine": {"type": "string", "enum": ["Italian", "Indian", "Chinese", "Mexican", "Vegan", "Japanese"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_table",
            "description": "Reserve a table at a restaurant",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_id": {"type": "integer", "description": "ID from recommendations"},
                    "time": {"type": "string", "description": "In HH:MM format"},
                    "guests": {"type": "integer", "description": "Number of people"}
                },
                "required": ["restaurant_id", "time", "guests"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reservation",
            "description": "Cancel an existing booking",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {"type": "integer", "description": "Booking ID"}
                },
                "required": ["reservation_id"]
            }
        }
    }
]

# ----- LLM Interaction ----- #
class RestaurantAgent:
    def __init__(self):
        try:
            self.llm_enabled = True  # Set False if API fails
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                raise ValueError("No token found in environment")
            self.client = InferenceClient(
                    provider="fireworks-ai",
                    api_key=token)
        except Exception as e:
            print(f"LLM disabled: {str(e)}")
            self.llm_enabled = False
    def process_query(self, user_input: str):
        """Main processing function with tool calling"""
        if not self.llm_enabled:
            return self._mock_response(user_input)
        #pdb.set_trace()
        # Handle weather queries directly through LLM
        if "weather" in user_input.lower():
            weather_prompt = f"Display the weather in {self._extract_location(user_input)} now in one line:"
            response = self._call_llm_weather(weather_prompt)
            return response
        
        # Normal tool-calling flow
        llm_response = self._call_llm(user_input, use_tools=True)
        
        # Handle tool calls
        """
        if isinstance(llm_response, dict) and "tool_calls" in llm_response and llm_response['tool_calls'] is not None:
            return self._execute_tool(llm_response["tool_calls"][0])
        """
        #pdb.set_trace()
        if isinstance(llm_response, dict) and "content" in llm_response:
            content = json.loads(llm_response['content']) 
            func_name = content["name"]
            params = content["parameters"]
            #params = json.loads(tool_call["function"]["arguments"])
            return self._execute_tool(func_name,params)
        
        return llm_response

    def _call_llm_weather(self, prompt: str, use_tools: bool = True):
        response = self.client.chat.completions.create(
            model= "meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            top_p=0.7
        )
        return response.choices[0].message


    def _call_llm(self, prompt: str, use_tools: bool = True):
        response = self.client.chat.completions.create(
            model= "meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            top_p=0.7,
            tools=TOOLS if use_tools else None,
            tool_choice="auto" if use_tools else None
        )
        #pdb.set_trace()
        #print(str(json.loads(response['choices'][0]['message']['content'])))
        return response.choices[0].message

    def _call_not_llm(self, prompt: str, use_tools: bool = True):
        """Simulate LLM API call (replace with actual implementation)"""
        if "weather" in prompt.lower():
            location = self._extract_location(prompt)
            return f"☀️ Current weather in {location}: Sunny, 28°C"  # Mock response
        
        # Mock tool-calling logic
        prompt = prompt.lower()
        
        if "recommend" in prompt or "find" in prompt:
            return {
                "tool_calls": [{
                    "function": {
                        "name": "get_restaurant_recommendations",
                        "arguments": json.dumps({
                            "location": self._extract_location(prompt),
                            "cuisine": next((c for c in ["Italian", "Indian", "Vegan"] if c in prompt), None)
                        })
                    }
                }]
            }
        elif "book" in prompt or "reserve" in prompt:
            return {
                "tool_calls": [{
                    "function": {
                        "name": "book_table",
                        "arguments": json.dumps({
                            "restaurant_id": random.randint(1, 25),
                            "time": "19:00",
                            "guests": 2
                        })
                    }
                }]
            }
        elif "cancel" in prompt:
            return {
                "tool_calls": [{
                    "function": {
                        "name": "cancel_reservation",
                        "arguments": json.dumps({
                            "reservation_id": 123  # Mock ID
                        })
                    }
                }]
            }
        else:
            return "Please specify if you want to book, cancel, or get recommendations."
    
    #def _execute_tool(self, tool_call: dict) -> str:
    def _execute_tool(self, func_name, params):
        """Execute the requested tool"""
        func_name = func_name
        params = params
        
        if func_name == "get_restaurant_recommendations":
            results = get_restaurant_recommendations(**params)
            return self._format_recommendations(results)
        elif func_name == "book_table":
            return book_table(**params)
        elif func_name == "cancel_reservation":
            return cancel_reservation(**params)
        else:
            return "Unknown command"
    
    def _extract_location(self, text: str) -> str:
        """Extract location from user input"""
        locations = ["MG Road", "Koramangala", "Whitefield", "Indiranagar", "HSR Layout"]
        return next((loc for loc in locations if loc.lower() in text.lower()), "Bengaluru")
    
    def _format_recommendations(self, restaurants: List[Dict]) -> str:
        if not restaurants:
            return "No restaurants match your criteria."
        return "🍽️ Recommended restaurants:\n" + "\n".join(
            f"- {r['name']} ({r['cuisine']}) in {r['location']} (Open: {r['open_time']}-{r['close_time']})"
            for r in restaurants[:3]  # Show top 3
        )
    
    def _mock_response(self, prompt: str) -> str:
        """Fallback when LLM is unavailable"""
        prompt = prompt.lower()
        if "book" in prompt:
            return "Mock: ✅ Booked table for 2 at 7 PM (ID: 123)"
        elif "cancel" in prompt:
            return "Mock: ✅ Cancelled reservation #123"
        elif "weather" in prompt:
            return f"Mock: ☀️ Current weather in {self._extract_location(prompt)}: Sunny, 28°C"
        elif "recommend" in prompt:
            return "Mock: 🍽️ Recommended: Spice Trail (Indian), Green Garden (Vegan)"
        else:
            return "Try: 'Book a table', 'Check weather in Koramangala', or 'Cancel reservation #123'"

# ----- Streamlit UI ----- #
def main():
    st.set_page_config(page_title="FoodieSpot AI", layout="wide")
    st.title("🍽️ FoodieSpot Reservation Agent")
    st.caption("Ask to book tables, check weather, or cancel reservations")
    
    init_db()
    agent = RestaurantAgent()
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I can help with:\n- Booking tables\n- Weather info\n- Cancellations"}
        ]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Try: 'Book a table', 'Weather in MG Road', or 'Cancel #123'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        response = agent.process_query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
