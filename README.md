# 🍽️ FoodieSpot AI Reservation Agent

*A conversational AI solution for restaurant bookings using meta-llama/Llama-3.3-70B-Instruct*

![Demo Screenshot](demo.gif) <!-- Add screenshot later -->

## Features
- Natural language table booking/modification
- Restaurant recommendations by cuisine/location
- Fallback to rule-based system when LLM unavailable
- SQLite database backend
- Streamlit web interface

## 🛠️ Setup

### Prerequisites
- Python 3.9+
- Hugging Face account (free)

```bash
# Clone repository
git clone https://github.com/shivaraj-arch/sarvam-demo.git
cd sarvam-demo

# Install dependencies
pip install -r requirements.txt

# Set up environment
export HUGGINGFACE_TOKEN="your_hf_token"  # Only needed for gated models

# Running the Application
streamlit run restaurant_agent.py

#Example usage
User: Book a table for 4 in Koramangala tonight
AI: ✅ Booked at Spice Trail (Koramangala) for 7 PM. Reservation ID: 42

weather will return the model completion response to tool call. the call can be subscribed to weather api.
