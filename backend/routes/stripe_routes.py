# from fastapi import APIRouter, HTTPException, Depends
# from typing import Optional
# import stripe
# from config import settings  # Assuming you have your stripe secret key in settings

# router = APIRouter()

# # Initialize Stripe with your secret key
# stripe.api_key = settings.STRIPE_SECRET_KEY

# @router.get("/verify-session")
# async def verify_session(session_id: str):
#     try:
#         # Retrieve the session from Stripe
#         session = stripe.checkout.Session.retrieve(session_id)
        
#         # Get the subscription details
#         subscription = stripe.Subscription.retrieve(session.subscription)
        
#         # Determine the plan type based on the price ID
#         price_id = subscription.plan.id
#         plan_type = 'pro' if price_id == settings.STRIPE_PRO_PRICE_ID else 'enterprise'
        
#         return {
#             "subscription": {
#                 "id": subscription.id,
#                 "plan": plan_type,
#                 "current_period_end": subscription.current_period_end,
#                 "status": subscription.status
#             }
#         }
#     except stripe.error.StripeError as e:
#         raise HTTPException(status_code=400, detail=str(e)) 