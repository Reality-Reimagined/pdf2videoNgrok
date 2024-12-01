import React, { useState, useEffect } from 'react';
import { useAuthStore } from '../../lib/store';
import { Check, Crown, Zap } from 'lucide-react';
import { toast } from 'react-hot-toast';
import { supabase } from '../../lib/supabase';

const plans = [
  {
    name: 'Free',
    price: 0,
    features: ['5 videos per month', 'Basic quality', 'Community support'],
    icon: Crown
  },
  {
    name: 'Pro',
    price: 29,
    checkoutUrl: 'https://buy.stripe.com/test_6oE2a93BO3q2bSw5kl',
    features: [
      'Unlimited videos',
      'HD quality',
      'Priority support',
      'Custom branding',
      'API access',
    ],
    icon: Zap,
    popular: true
  },
  {
    name: 'Enterprise',
    price: 99,
    checkoutUrl: 'https://buy.stripe.com/test_7sI9CB8W8f8K4q4bIK',
    features: [
      'Everything in Pro',
      '4K quality',
      'Dedicated support',
      'Custom integrations',
      'Team collaboration',
    ],
    icon: Crown
  },
];

export const SubscriptionManager = () => {
  const { user } = useAuthStore();
  const [currentPlan, setCurrentPlan] = useState('Free');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkSubscription = async () => {
      if (!user?.id) {
        console.log('No user ID found');
        setIsLoading(false);
        return;
      }

      try {
        console.log('Checking subscription for user:', user.id);

        const { data, error } = await supabase
          .from('user_subscriptions')
          .select('*')  // Select all columns for debugging
          .eq('user_id', user.id);

        console.log('Raw subscription data:', data);
        console.log('Query error:', error);

        if (error) {
          console.error('Subscription check error:', error);
          return;
        }

        if (data && data.length > 0) {
          // Find active subscription
          const activeSubscription = data.find(sub => 
            sub.status === 'active' && 
            (sub.plan_type === 'pro' || sub.plan_type === 'enterprise')
          );

          console.log('Active subscription found:', activeSubscription);

          if (activeSubscription) {
            const plan = activeSubscription.plan_type.toLowerCase();
            console.log('Setting plan to:', plan);
            setCurrentPlan(plan.charAt(0).toUpperCase() + plan.slice(1));
          } else {
            console.log('No active pro/enterprise subscription found');
            setCurrentPlan('Free');
          }
        } else {
          console.log('No subscriptions found for user');
          setCurrentPlan('Free');
        }
      } catch (error) {
        console.error('Subscription check failed:', error);
      } finally {
        setIsLoading(false);
      }
    };

    checkSubscription();
  }, [user]);

  const handleSubscribe = (checkoutUrl?: string) => {
    if (!user) {
      toast.error('Please login to subscribe');
      return;
    }

    if (checkoutUrl) {
      window.location.href = checkoutUrl;
    }
  };

  if (isLoading) {
    return (
      <div className="py-12 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <p>Loading subscription status...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="py-12 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
            Choose your plan
          </h2>
          <p className="mt-4 text-lg text-gray-600">
            Select the perfect plan for your needs
          </p>
          {currentPlan !== 'Free' && (
            <p className="mt-2 text-sm text-blue-600">
              Current subscription: {currentPlan}
            </p>
          )}
        </div>

        <div className="mt-12 space-y-4 sm:mt-16 sm:space-y-0 sm:grid sm:grid-cols-2 sm:gap-6 lg:max-w-4xl lg:mx-auto xl:max-w-none xl:grid-cols-3">
          {plans.map((plan) => {
            const Icon = plan.icon;
            const isCurrentPlan = plan.name.toLowerCase() === currentPlan.toLowerCase();
            
            return (
              <div
                key={plan.name}
                className={`rounded-lg shadow-lg divide-y divide-gray-200 bg-white
                  ${plan.popular ? 'ring-2 ring-blue-500' : ''}
                  ${isCurrentPlan ? 'ring-2 ring-green-500' : ''}
                `}
              >
                <div className="p-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-2xl font-semibold text-gray-900">
                      {plan.name}
                    </h3>
                    <Icon className={`h-6 w-6 ${
                      plan.popular ? 'text-blue-500' : 'text-gray-400'
                    }`} />
                  </div>
                  
                  <p className="mt-8">
                    <span className="text-4xl font-extrabold text-gray-900">
                      ${plan.price}
                    </span>
                    <span className="text-base font-medium text-gray-500">
                      /month
                    </span>
                  </p>

                  <button
                    onClick={() => handleSubscribe(plan.checkoutUrl)}
                    disabled={isCurrentPlan}
                    className={`mt-8 block w-full py-3 px-6 rounded-md font-semibold
                      ${isCurrentPlan 
                        ? 'bg-green-600 text-white cursor-not-allowed'
                        : 'bg-blue-600 hover:bg-blue-700 text-white'
                      }
                    `}
                  >
                    {isCurrentPlan ? 'Current Plan' : `Subscribe to ${plan.name}`}
                  </button>
                </div>

                <div className="px-6 pt-6 pb-8">
                  <h4 className="text-sm font-semibold text-gray-900 tracking-wide uppercase">
                    What's included
                  </h4>
                  <ul className="mt-6 space-y-4">
                    {plan.features.map((feature) => (
                      <li key={feature} className="flex space-x-3">
                        <Check className="flex-shrink-0 h-5 w-5 text-green-500" />
                        <span className="text-base text-gray-700">{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};