# PDF2Video with Stripe and Supabase

## Setup
1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your values
3. Install dependencies:
   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt

   # Frontend
   cd frontend
   npm install
   ```

## Development
1. Start the backend server:
   ```bash
   cd backend
   uvicorn routes.main:app --reload
   ```
2. Start the frontend server:
   ```bash
   cd frontend
   npm run dev
   ```

