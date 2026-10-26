# AI Canteen Connect

A full-stack food ordering app for a college canteen (React, Node.js, MongoDB), plus a separate Python service with a hybrid food recommender and an hourly demand forecast.

## What works
- **Customer app (React + Vite):** browse the menu by category, add to cart, sign up / log in (JWT), place an order, pay by Stripe Checkout or cash on delivery, and see your orders in "My Orders".
- **Admin panel (React):** add and remove menu items with images, and see all orders and update each order's status (for example "Food Processing" to "Delivered").
- **Backend (Express + MongoDB):** REST API for food, users, cart and orders, with password hashing (bcrypt) and token-based auth.
- **ML service (Flask, `ml-service/`):** runs on its own on port 8000.
  - `/recommend`: hybrid recommender = 0.6 x collaborative item-item similarity + 0.4 x TF-IDF content similarity.
  - `/forecast`: XGBoost model that predicts items ordered in an hour for a category.

## Honest status
- The ML service is **not connected to the customer app yet**. It is a working standalone service.
- It was built and tested on **generated sample data** (no real canteen orders). On that data the demand model scores about 60% (1 - WAPE), which only shows the pipeline works, not real-world performance.
- Order status changes appear when the user reloads "My Orders"; there are no WebSockets.
- No real-user, uptime or wait-time results are claimed.

## Screenshots
![Menu](screenshots/menu.png)
![Checkout](screenshots/checkout.png)
![Order tracking](screenshots/order_tracking.png)
![Admin panel](screenshots/Admin%20panel.png)

## Tech stack
React 18, Vite, React Router, Axios, Node.js, Express, MongoDB (Mongoose), JWT, bcrypt, Stripe, Flask, scikit-learn, XGBoost, pandas.

## Run it
```bash
# backend: create backend/.env with MONGODB_URI, JWT_SECRET and STRIPE_SECRET_KEY
cd backend && npm install && npm run server      # http://localhost:4000 (see server.js)

# customer app
cd frontend && npm install && npm run dev        # http://localhost:5173

# admin panel
cd admin && npm install && npm run dev

# ML service
cd ml-service && pip install -r requirements.txt && python app.py    # http://localhost:8000
```
Or run the backend and ML service together with Docker: copy `backend/.env.example` to `backend/.env`, fill it in, then `docker compose up --build` (the Dockerfiles and compose file have not been tested on a cloud server).

See `ml-service/README.md` for the ML endpoints.

## Next steps
- Call the ML service from the menu page to show recommendations.
- Replace the sample data with real order data and re-measure.
- Add live order updates.

## Author
Kaveri Anil Ghatage | LinkedIn: linkedin.com/in/kaverighatage | GitHub: Kaveri-Ghatage

Educational project.
